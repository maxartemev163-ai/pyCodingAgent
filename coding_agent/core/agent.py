"""Main coding agent implementation."""

import json
import logging
from typing import Optional

from ..config import ModelConfig, Settings
from ..llm import LLMClient, Message, Role
from ..llm.message import ToolCall
from ..tools import ToolRegistry, ToolResult
from .context import ConversationContext

logger = logging.getLogger(__name__)


class CodingAgent:
    """Main coding agent that orchestrates LLM interactions and tool execution.

    The agent follows a ReAct (Reasoning + Acting) pattern:
    1. Receive user input
    2. Query LLM for response/tool calls
    3. Execute tools if requested
    4. Feed results back to LLM
    5. Repeat until completion

    Attributes:
        settings: Agent configuration settings.
        model_config: LLM model configuration.
        tool_registry: Registry of available tools.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        model_config: Optional[ModelConfig] = None,
        tool_registry: Optional[ToolRegistry] = None,
        prepare_context: bool = True,
    ) -> None:
        """Initialize the coding agent.

        Args:
            settings: Agent settings. Uses defaults if not provided.
            model_config: LLM configuration. Uses defaults if not provided.
            tool_registry: Tool registry. Creates empty registry if not provided.
            prepare_context: Whether to prepare session context before starting.
        """
        self.settings = settings or Settings()
        self.model_config = model_config or ModelConfig()
        self.tool_registry = tool_registry or ToolRegistry()

        # Prepare session context if requested
        session_context_str = None
        if prepare_context:
            from .session_context import prepare_session_context
            from .skills_loader import load_skills_context

            session_ctx = prepare_session_context(
                workspace_dir=self.settings.workspace_dir,
                requirements_file="requirements.txt",
            )
            session_context_str = session_ctx.to_system_prompt()
            
            # Load skills/rules from Markdown files and append to session context
            skills_context = load_skills_context(workspace_dir=self.settings.workspace_dir)
            if skills_context:
                session_context_str = session_context_str + "\n\n" + skills_context

        self._client = LLMClient(self.model_config)
        self._context = ConversationContext(
            max_length=self.settings.max_iterations,
            system_prompt=self._get_default_system_prompt(),
            session_context=session_context_str,
        )

        if self.settings.enable_history:
            history_path = Path(self.settings.workspace_dir) / self.settings.history_dir
            self._context.history_file = history_path / "conversation.json"

    def _get_default_system_prompt(self) -> str:
        """Generate the default system prompt for the agent.

        Returns:
            System prompt string.
        """
        return """You are an expert coding assistant. You help users with software development tasks.

CRITICAL RULES - READ CAREFULLY:
1. NEVER just describe how to do something - ALWAYS USE TOOLS to actually DO it
2. When user asks to open/read a file, you MUST use the 'read_file' tool - do NOT tell them to use cat or any command
3. When user asks to create/write a file, you MUST use the 'write_file' tool
4. You CANNOT complete tasks without using tools - describing actions is NOT completing the task

Guidelines:
- Think step by step before taking action
- Use available tools to accomplish EVERY task
- Always verify your work when possible
- Write clean, well-documented code
- Follow best practices and design patterns
- If unsure, ask clarifying questions
- Prefer reading existing files before modifying them
- Test changes when appropriate

Tool Usage Instructions:
- When you need to perform ANY action, you MUST use the tool_calls format in your response
- Do NOT write tool calls as JSON text in your message content
- The system will automatically parse your tool calls and execute them
- After tools execute, you will see their results and can continue
- NEVER say "you can use X command" - YOU are the one who executes, use tools!

Available tools allow you to:
- read_file: Read contents of a file (USE THIS when user says "open", "show", "read", "view" a file)
- write_file: Write content to a file
- list_dir: List contents of a directory
- search_files: Search for files matching a glob pattern
- get_tree: Get a tree view of files and directories
- run_command: Execute shell commands
- run_python: Run Python code snippets

Always explain what you're doing and why.

Examples of correct tool usage:

Example 1 - Creating a file:
User: Create main.py with a hello world program
Assistant: [Uses write_file tool]
  Tool: write_file
  Arguments: {"path": "main.py", "content": "print('Hello, World!')"}

Example 2 - Reading a file then modifying:
User: Update the greeting in app.py
Assistant: [First uses read_file to check current content]
  Tool: read_file
  Arguments: {"path": "app.py"}
[After seeing file content]
Assistant: [Uses write_file to update]
  Tool: write_file
  Arguments: {"path": "app.py", "content": "print('Welcome!')"}

Example 3 - User wants to see a file:
User: open app.py and show data
Assistant: [IMMEDIATELY uses read_file tool - does NOT explain how user can do it]
  Tool: read_file
  Arguments: {"path": "app.py"}

Example 4 - Running a command:
User: List all Python files in the current directory
Assistant: [Uses run_command tool]
  Tool: run_command
  Arguments: {"command": "dir *.py"}

Example 5 - Getting a tree view:
User: Show me the directory structure
Assistant: [Uses get_tree tool]
  Tool: get_tree
  Arguments: {"path": ".", "max_depth": 2}

REMEMBER: 
- If user says "open file X", USE read_file tool with path="X"
- If user says "show file X", USE read_file tool with path="X"  
- NEVER tell the user how THEY can do something - YOU do it with tools
- Always use tool calls for actions, never output JSON directly in your response text."""

    def run(self, user_input: str, stream: bool = False) -> str:
        """Process a user request and return the agent's response.

        Args:
            user_input: The user's request or question.
            stream: Whether to stream the response (not yet implemented).

        Returns:
            The agent's final response.
        """
        self._context.add_user_message(user_input)
        logger.info(f"Processing user request: {user_input[:50]}...")

        iteration = 0
        tool_call_history: list[str] = []  # Track tool calls to detect loops
        
        while iteration < self.settings.max_iterations:
            iteration += 1
            logger.info(f"Iteration {iteration}/{self.settings.max_iterations}")

            messages = self._context.get_messages()
            tools_schema = self.tool_registry.get_all_schemas()

            try:
                logger.info("Waiting for LLM response...")
                content, tool_calls = self._client.chat(messages, tools=tools_schema)
                logger.info(f"LLM response received (content length: {len(content)}, tool calls: {len(tool_calls) if tool_calls else 0})")
            except Exception as e:
                logger.error(f"LLM request failed: {e}")
                return f"Error: Failed to communicate with LLM - {e}"

            # Execute tool calls if present
            if tool_calls:
                logger.info(f"Executing {len(tool_calls)} tool call(s)")
                
                # Check for repeated tool calls (infinite loop detection)
                for tool_call in tool_calls:
                    call_signature = f"{tool_call.name}:{tool_call.arguments}"
                    
                    # Count how many times this exact call was made
                    occurrences = tool_call_history.count(call_signature)
                    
                    if occurrences >= 2:
                        # Detected repetition - break the loop
                        logger.warning(
                            f"Detected repeated tool call pattern: '{call_signature}' "
                            f"(occurred {occurrences + 1} times). Breaking loop to prevent infinite cycling."
                        )
                        self._context.add_assistant_message(
                            "I notice I'm repeating the same action. Let me try a different approach or ask for clarification.\n\n"
                            f"What I've tried multiple times: {tool_call.name}\n"
                            "Suggestion: Please provide more specific details about what you'd like me to do differently."
                        )
                        self._context._save_history()
                        return "I noticed I was repeating the same action. Could you please clarify what you'd like me to do?"
                    
                    result = self._execute_tool_call(tool_call)
                    self._context.add_tool_result(tool_call.id, result.output)
                    logger.info(f"Tool '{tool_call.name}' executed: success={result.success}")

                    if not result.success:
                        logger.warning(f"Tool execution failed: {result.error}")
                    
                    # Print tool output to console for user visibility
                    # Always print tool results except when explicitly marked as "(no output)"
                    if result.output != "(no output)":
                        print(f"\n[Tool: {tool_call.name}]")
                        if result.output:
                            print(result.output)
                        else:
                            print("(empty output)")
                        if not result.success and result.error:
                            print(f"[Error: {result.error}]")
                    
                    # Add to history for loop detection
                    tool_call_history.append(call_signature)
                
                # Continue the loop to let LLM process tool results
                continue

            # No tool calls, we're done
            if content:
                self._context.add_assistant_message(content)
            
            self._context._save_history()
            return content

        logger.warning("Max iterations reached")
        return "I've reached the maximum number of iterations. Let me summarize what I've accomplished..."

    def _execute_tool_call(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call from the LLM.

        Args:
            tool_call: The tool call to execute.

        Returns:
            Result of the tool execution.
        """
        try:
            arguments = json.loads(tool_call.arguments) if tool_call.arguments else {}
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON arguments for tool '{tool_call.name}': {tool_call.arguments}")
            return ToolResult(success=False, error="Invalid JSON arguments")

        logger.info(f"Executing tool: {tool_call.name} with args: {arguments}")

        try:
            result = self.tool_registry.execute(tool_call.name, **arguments)
            logger.info(f"Tool '{tool_call.name}' completed: success={result.success}")
            return result
        except KeyError:
            logger.error(f"Unknown tool requested: {tool_call.name}")
            return ToolResult(success=False, error=f"Unknown tool: {tool_call.name}")
        except Exception as e:
            logger.error(f"Tool '{tool_call.name}' execution error: {e}")
            return ToolResult(success=False, error=f"Tool execution error: {e}")

    def register_tool(self, tool) -> None:
        """Register a tool with the agent.

        Args:
            tool: Tool instance to register.
        """
        self.tool_registry.register(tool)
        logger.info(f"Registered tool: {tool.name}")

    def get_context(self) -> ConversationContext:
        """Get the current conversation context.

        Returns:
            The conversation context.
        """
        return self._context

    def clear_context(self) -> None:
        """Clear the conversation history."""
        self._context.clear()
        logger.info("Cleared conversation context")

    def retry_last_task(self) -> Optional[str]:
        """Retry the last task by removing it from history and re-processing.

        This method removes the last user message and any subsequent messages
        (assistant responses and tool results), then returns the removed user
        message content so it can be re-processed.

        Returns:
            The content of the last user message that was removed, or None if
            no user message was found to retry.
        """
        last_message = self._context.get_last_user_message()
        if last_message is None:
            logger.warning("No user message found to retry")
            return None

        if self._context.remove_last_user_message():
            logger.info(f"Retrying last task: {last_message[:50]}...")
            return last_message
        else:
            logger.warning("Failed to remove last user message")
            return None

    def close(self) -> None:
        """Close the agent and release resources."""
        self._client.close()
        logger.info("Agent closed")

    def __enter__(self) -> "CodingAgent":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


# Import Path here to avoid circular imports
from pathlib import Path

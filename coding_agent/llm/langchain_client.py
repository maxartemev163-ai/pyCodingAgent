"""LangChain-based LLM client with comprehensive logging."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool as langchain_tool
from langchain_openai import ChatOpenAI

from ..config import ModelConfig
from .message import Message, Role, ToolCall

logger = logging.getLogger(__name__)


class LangChainClient:
    """LangChain-based client for OpenAI-compatible LLM APIs.

    This client uses LangChain's ChatOpenAI to interact with local models
    like Qwen2.5-Coder via Ollama or similar services.

    Attributes:
        config: Model configuration settings.
        logs_dir: Directory path for storing request/response logs.
    """

    def __init__(self, config: ModelConfig, logs_dir: Optional[str] = None) -> None:
        """Initialize the LangChain client.

        Args:
            config: Model configuration.
            logs_dir: Optional directory for storing LLM request/response logs.
                     If None, defaults to 'logs' directory in current working directory.
        """
        self.config = config
        
        # Set up logs directory
        if logs_dir is None:
            logs_dir = Path.cwd() / "logs"
        self.logs_dir = Path(logs_dir)
        self._ensure_logs_directory()
        
        # Initialize LangChain ChatOpenAI client
        self._client = ChatOpenAI(
            model=config.model_name,
            base_url=config.base_url,
            api_key=config.api_key,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
        )
        
        logger.info(f"LangChain client initialized for model: {config.model_name}")
        logger.info(f"Logs will be saved to: {self.logs_dir}")

    def _ensure_logs_directory(self) -> None:
        """Create logs directory if it doesn't exist."""
        try:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Logs directory ensured: {self.logs_dir}")
        except Exception as e:
            logger.warning(f"Failed to create logs directory: {e}")

    def _log_request_response(
        self,
        messages: list[BaseMessage],
        response: AIMessage,
        tools: Optional[list[dict]] = None,
        duration: float = 0.0,
    ) -> None:
        """Save request and response to log file.

        Args:
            messages: List of conversation messages sent to LLM.
            response: Response received from LLM.
            tools: Optional list of tool schemas.
            duration: Request duration in seconds.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_file = self.logs_dir / f"llm_interaction_{timestamp}.json"
        
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "model": self.config.model_name,
            "request": {
                "messages": [self._message_to_dict(msg) for msg in messages],
                "tools": tools,
                "parameters": {
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "max_tokens": self.config.max_tokens,
                },
            },
            "response": {
                "content": response.content,
                "tool_calls": self._extract_tool_calls(response),
                "metadata": {
                    "usage_metadata": getattr(response, "usage_metadata", None),
                    "response_metadata": getattr(response, "response_metadata", None),
                },
            },
            "duration_seconds": duration,
        }
        
        try:
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2, default=str)
            logger.debug(f"Logged interaction to {log_file}")
        except Exception as e:
            logger.error(f"Failed to save log file: {e}")

    def _message_to_dict(self, message: BaseMessage) -> dict:
        """Convert LangChain message to dictionary.

        Args:
            message: LangChain message object.

        Returns:
            Dictionary representation of the message.
        """
        result = {
            "role": message.type,
            "content": message.content,
        }
        
        if isinstance(message, AIMessage) and message.tool_calls:
            result["tool_calls"] = message.tool_calls
            
        if isinstance(message, ToolMessage):
            result["tool_call_id"] = message.tool_call_id
            
        return result

    def _extract_tool_calls(self, response: AIMessage) -> list[dict]:
        """Extract tool calls from LangChain response.

        Args:
            response: AI message response.

        Returns:
            List of tool call dictionaries.
        """
        tool_calls = []
        
        # LangChain stores tool_calls in different ways depending on version
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                tool_calls.append({
                    "id": tc.get("id", ""),
                    "name": tc.get("name", ""),
                    "arguments": tc.get("args", {}),
                })
                
        return tool_calls

    def _convert_messages(
        self,
        messages: list[Message],
    ) -> list[BaseMessage]:
        """Convert internal Message format to LangChain messages.

        Args:
            messages: List of internal Message objects.

        Returns:
            List of LangChain BaseMessage objects.
        """
        langchain_messages = []
        
        for msg in messages:
            if msg.role == Role.SYSTEM:
                langchain_messages.append(SystemMessage(content=msg.content))
            elif msg.role == Role.USER:
                langchain_messages.append(HumanMessage(content=msg.content))
            elif msg.role == Role.ASSISTANT:
                langchain_messages.append(AIMessage(content=msg.content))
            elif msg.role == Role.TOOL:
                langchain_messages.append(
                    ToolMessage(
                        content=msg.content,
                        tool_call_id=msg.tool_call_id or "",
                    )
                )
        
        return langchain_messages

    def chat(
        self,
        messages: list[Message],
        tools: Optional[list[dict]] = None,
        stream: bool = False,
    ) -> tuple[str, Optional[list[ToolCall]]]:
        """Send a chat request to the LLM using LangChain.

        Args:
            messages: List of conversation messages.
            tools: Optional list of tool schemas.
            stream: Whether to stream the response (not yet implemented).

        Returns:
            Tuple of (response content, optional tool calls).
        """
        import time
        
        langchain_messages = self._convert_messages(messages)
        
        logger.info(f"Sending chat request to LLM (model: {self.config.model_name})")
        logger.debug(f"Messages count: {len(langchain_messages)}")
        
        # Convert tools to LangChain format if provided
        langchain_tools = None
        if tools:
            langchain_tools = self._convert_tools_to_langchain(tools)
        
        start_time = time.time()
        
        try:
            # Make the API call through LangChain
            if langchain_tools:
                response = self._client.bind_tools(langchain_tools).invoke(
                    langchain_messages
                )
            else:
                response = self._client.invoke(langchain_messages)
            
            duration = time.time() - start_time
            
            # Log the interaction
            self._log_request_response(
                messages=langchain_messages,
                response=response,
                tools=tools,
                duration=duration,
            )
            
            # Parse the response
            content = response.content if response.content else ""
            tool_calls = self._parse_langchain_tool_calls(response)
            
            logger.info(f"LLM response received successfully (duration: {duration:.2f}s)")
            logger.debug(f"Response content length: {len(content)}, tool_calls: {len(tool_calls) if tool_calls else 0}")
            
            return content, tool_calls
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"LLM request failed after {duration:.2f}s: {e}")
            raise RuntimeError(f"Failed to communicate with LLM: {e}") from e

    def _convert_tools_to_langchain(self, tools: list[dict]) -> list:
        """Convert tool schemas to LangChain tools.

        Args:
            tools: List of tool schema dictionaries.

        Returns:
            List of LangChain tool objects.
        """
        # For now, we pass the raw schemas as LangChain can work with them
        # In a more advanced implementation, we could create actual @tool decorators
        return tools

    def _parse_langchain_tool_calls(
        self,
        response: AIMessage,
    ) -> Optional[list[ToolCall]]:
        """Parse tool calls from LangChain response.

        Args:
            response: LangChain AI message response.

        Returns:
            List of ToolCall objects if found, None otherwise.
        """
        if not hasattr(response, "tool_calls") or not response.tool_calls:
            return None
        
        tool_calls = []
        for i, tc in enumerate(response.tool_calls):
            tool_call = ToolCall(
                id=tc.get("id", f"call_{i}"),
                name=tc.get("name", ""),
                arguments=json.dumps(tc.get("args", {})),
            )
            tool_calls.append(tool_call)
        
        return tool_calls if tool_calls else None

    def close(self) -> None:
        """Close the client and release resources."""
        logger.info("LangChain client closed")

    def __enter__(self) -> "LangChainClient":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()

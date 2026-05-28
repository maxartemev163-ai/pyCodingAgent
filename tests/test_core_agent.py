"""Tests for the main coding agent implementation."""

import json
from unittest.mock import MagicMock, patch

import pytest

from coding_agent.core.agent import CodingAgent
from coding_agent.config import ModelConfig, Settings
from coding_agent.llm.message import Message, Role, ToolCall
from coding_agent.tools import ToolRegistry, ToolResult
from coding_agent.tools.base import Tool


class MockTool(Tool):
    """Mock tool for testing."""

    @property
    def name(self) -> str:
        return "mock_tool"

    @property
    def description(self) -> str:
        return "A mock tool for testing"

    @property
    def schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "value": {"type": "string", "description": "A test value"},
            },
        }

    def execute(self, **kwargs) -> ToolResult:
        value = kwargs.get("value", "default")
        return ToolResult(success=True, output=f"Mock executed with value: {value}")


class TestCodingAgentInitialization:
    """Tests for CodingAgent initialization."""

    def test_init_with_defaults(self):
        """Test initializing agent with default settings."""
        agent = CodingAgent()

        assert isinstance(agent.settings, Settings)
        assert isinstance(agent.model_config, ModelConfig)
        assert isinstance(agent.tool_registry, ToolRegistry)
        assert agent._context is not None

    def test_init_with_custom_settings(self):
        """Test initializing agent with custom settings."""
        settings = Settings(max_iterations=5, enable_history=False)
        model_config = ModelConfig(model_name="test-model")
        tool_registry = ToolRegistry()

        agent = CodingAgent(
            settings=settings,
            model_config=model_config,
            tool_registry=tool_registry,
        )

        assert agent.settings.max_iterations == 5
        assert agent.model_config.model_name == "test-model"
        assert agent.tool_registry is tool_registry

    def test_get_default_system_prompt(self):
        """Test that default system prompt contains expected content."""
        agent = CodingAgent()
        prompt = agent._get_default_system_prompt()

        assert "expert coding assistant" in prompt.lower()
        assert "Think step by step" in prompt
        assert "Available tools" in prompt


class TestCodingAgentRun:
    """Tests for CodingAgent run method."""

    @patch("coding_agent.core.agent.LangChainClient")
    def test_run_without_tool_calls(self, mock_llm_client_class):
        """Test running agent when LLM returns no tool calls."""
        mock_client = MagicMock()
        mock_client.chat.return_value = ("Hello! How can I help?", None)
        mock_llm_client_class.return_value = mock_client

        agent = CodingAgent()
        result = agent.run("Hello")

        assert result == "Hello! How can I help?"
        mock_client.chat.assert_called_once()

    @patch("coding_agent.core.agent.LangChainClient")
    def test_run_with_tool_calls(self, mock_llm_client_class):
        """Test running agent when LLM returns tool calls."""
        mock_client = MagicMock()
        # First call returns tool call, second returns final response
        tool_call = ToolCall(
            id="call_1",
            name="mock_tool",
            arguments=json.dumps({"value": "test"}),
        )
        mock_client.chat.side_effect = [
            ("Thinking...", [tool_call]),
            ("Done!", None),
        ]
        mock_llm_client_class.return_value = mock_client

        agent = CodingAgent()
        agent.register_tool(MockTool())
        result = agent.run("Do something")

        assert result == "Done!"
        assert mock_client.chat.call_count == 2

    @patch("coding_agent.core.agent.LangChainClient")
    def test_run_handles_llm_error(self, mock_llm_client_class):
        """Test that LLM errors are handled gracefully."""
        mock_client = MagicMock()
        mock_client.chat.side_effect = Exception("Connection failed")
        mock_llm_client_class.return_value = mock_client

        agent = CodingAgent()
        result = agent.run("Hello")

        assert "Error" in result
        assert "Failed to communicate with LLM" in result

    @patch("coding_agent.core.agent.LangChainClient")
    def test_run_max_iterations_reached(self, mock_llm_client_class):
        """Test that max iterations limit is enforced."""
        mock_client = MagicMock()
        # Always return tool calls to trigger max iterations
        tool_call = ToolCall(
            id="call_1",
            name="mock_tool",
            arguments=json.dumps({}),
        )
        mock_client.chat.return_value = ("Thinking...", [tool_call])
        mock_llm_client_class.return_value = mock_client

        settings = Settings(max_iterations=2)
        agent = CodingAgent(settings=settings)
        agent.register_tool(MockTool())

        result = agent.run("Infinite loop task")

        assert "maximum number of iterations" in result

    @patch("coding_agent.core.agent.LangChainClient")
    def test_run_invalid_json_arguments(self, mock_llm_client_class):
        """Test handling of invalid JSON in tool arguments."""
        mock_client = MagicMock()
        tool_call = ToolCall(
            id="call_1",
            name="mock_tool",
            arguments="not valid json{{{",
        )
        mock_client.chat.return_value = ("Thinking...", [tool_call])
        mock_llm_client_class.return_value = mock_client

        agent = CodingAgent()
        agent.register_tool(MockTool())
        result = agent.run("Test")

        # Should continue and eventually return or hit max iterations
        # The invalid JSON should be handled gracefully
        assert result is not None


class TestCodingAgentToolExecution:
    """Tests for tool execution in CodingAgent."""

    @patch("coding_agent.core.agent.LangChainClient")
    def test_execute_tool_call_success(self, mock_llm_client_class):
        """Test successful tool execution."""
        mock_client = MagicMock()
        tool_call = ToolCall(
            id="call_1",
            name="mock_tool",
            arguments=json.dumps({"value": "success"}),
        )
        # First call returns tool call, second returns final response
        mock_client.chat.side_effect = [
            ("Thinking...", [tool_call]),
            ("Done! Mock executed with value: success", None),
        ]
        mock_llm_client_class.return_value = mock_client

        agent = CodingAgent()
        agent.register_tool(MockTool())
        result = agent.run("Execute tool")

        assert "Mock executed" in result or "Done!" in result

    @patch("coding_agent.core.agent.LangChainClient")
    def test_execute_unknown_tool(self, mock_llm_client_class):
        """Test execution of unknown tool."""
        mock_client = MagicMock()
        tool_call = ToolCall(
            id="call_1",
            name="unknown_tool",
            arguments=json.dumps({}),
        )
        mock_client.chat.return_value = ("Calling tool...", [tool_call])
        mock_llm_client_class.return_value = mock_client

        agent = CodingAgent()
        result = agent.run("Test unknown tool")

        # Should handle unknown tool gracefully
        assert result is not None

    @patch("coding_agent.core.agent.LangChainClient")
    def test_execute_tool_with_exception(self, mock_llm_client_class):
        """Test tool execution that raises an exception."""
        mock_client = MagicMock()
        tool_call = ToolCall(
            id="call_1",
            name="mock_tool",
            arguments=json.dumps({}),
        )
        mock_client.chat.return_value = ("Calling...", [tool_call])
        mock_llm_client_class.return_value = mock_client

        # Create a mock tool that raises an exception
        class FailingTool(Tool):
            @property
            def name(self) -> str:
                return "mock_tool"

            @property
            def description(self) -> str:
                return "A failing tool"

            @property
            def schema(self) -> dict:
                return {"type": "object", "properties": {}}

            def execute(self, **kwargs) -> ToolResult:
                raise RuntimeError("Tool execution failed")

        agent = CodingAgent()
        agent.register_tool(FailingTool())
        result = agent.run("Test failing tool")

        # Should handle exception gracefully
        assert result is not None


class TestCodingAgentContextManagement:
    """Tests for context management in CodingAgent."""

    def test_get_context(self):
        """Test getting the conversation context."""
        agent = CodingAgent()
        context = agent.get_context()

        assert context is not None
        from coding_agent.core.context import ConversationContext

        assert isinstance(context, ConversationContext)

    def test_clear_context(self):
        """Test clearing the conversation context."""
        agent = CodingAgent()
        context = agent.get_context()
        context.add_user_message("Test message")

        assert len(context.messages) > 0

        agent.clear_context()

        # After clearing, should only have system prompt if any
        cleared_context = agent.get_context()
        assert len(cleared_context.messages) == 0 or cleared_context.messages[0].role == Role.SYSTEM

    @patch("coding_agent.core.agent.LangChainClient")
    def test_close_agent(self, mock_llm_client_class):
        """Test closing the agent."""
        mock_client = MagicMock()
        mock_llm_client_class.return_value = mock_client

        agent = CodingAgent()
        agent.close()

        mock_client.close.assert_called_once()

    def test_context_manager_enter_exit(self):
        """Test using agent as a context manager."""
        with patch("coding_agent.core.agent.LangChainClient") as mock_llm_client_class:
            mock_client = MagicMock()
            mock_llm_client_class.return_value = mock_client

            with CodingAgent() as agent:
                assert isinstance(agent, CodingAgent)

            mock_client.close.assert_called_once()


class TestCodingAgentRegistration:
    """Tests for tool registration."""

    def test_register_tool(self):
        """Test registering a tool with the agent."""
        agent = CodingAgent()
        tool = MockTool()

        initial_count = len(agent.tool_registry._tools)
        agent.register_tool(tool)
        final_count = len(agent.tool_registry._tools)

        assert final_count == initial_count + 1
        assert "mock_tool" in agent.tool_registry._tools

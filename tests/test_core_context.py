"""Tests for the conversation context management."""

import json
import tempfile
from pathlib import Path

import pytest

from coding_agent.core.context import ConversationContext
from coding_agent.llm.message import Message, Role


class TestConversationContext:
    """Tests for the ConversationContext class."""

    def test_create_empty_context(self):
        """Test creating an empty conversation context."""
        ctx = ConversationContext()
        assert ctx.messages == []
        assert ctx.max_length == 100
        assert ctx.system_prompt is None
        assert ctx.history_file is None

    def test_create_context_with_system_prompt(self):
        """Test creating a context with a system prompt."""
        ctx = ConversationContext(system_prompt="You are a helpful assistant.")
        assert len(ctx.messages) == 1
        assert ctx.messages[0].role == Role.SYSTEM
        assert ctx.messages[0].content == "You are a helpful assistant."

    def test_add_user_message(self):
        """Test adding a user message."""
        ctx = ConversationContext()
        ctx.add_user_message("Hello")

        assert len(ctx.messages) == 1
        assert ctx.messages[0].role == Role.USER
        assert ctx.messages[0].content == "Hello"

    def test_add_assistant_message(self):
        """Test adding an assistant message."""
        ctx = ConversationContext()
        ctx.add_assistant_message("Hi there!")

        assert len(ctx.messages) == 1
        assert ctx.messages[0].role == Role.ASSISTANT
        assert ctx.messages[0].content == "Hi there!"

    def test_add_tool_result(self):
        """Test adding a tool result message."""
        ctx = ConversationContext()
        ctx.add_tool_result("call_123", "Tool output here")

        assert len(ctx.messages) == 1
        assert ctx.messages[0].role == Role.TOOL
        assert ctx.messages[0].content == "Tool output here"
        assert ctx.messages[0].tool_call_id == "call_123"

    def test_get_messages_returns_copy(self):
        """Test that get_messages returns a copy, not the original list."""
        ctx = ConversationContext()
        ctx.add_user_message("Test")

        messages = ctx.get_messages()
        messages.append(Message(role=Role.USER, content="Another"))

        # Original context should be unchanged
        assert len(ctx.messages) == 1

    def test_clear_removes_all_messages(self):
        """Test that clear removes all messages except system prompt."""
        ctx = ConversationContext(system_prompt="System prompt")
        ctx.add_user_message("User message")
        ctx.add_assistant_message("Assistant message")

        ctx.clear()

        # Should only have system message left
        assert len(ctx.messages) == 1
        assert ctx.messages[0].role == Role.SYSTEM

    def test_clear_without_system_prompt(self):
        """Test that clear works without a system prompt."""
        ctx = ConversationContext()
        ctx.add_user_message("User message")

        ctx.clear()

        assert ctx.messages == []

    def test_trim_if_needed_trims_excess_messages(self):
        """Test that messages are trimmed when exceeding max_length."""
        ctx = ConversationContext(max_length=5)
        for i in range(10):
            ctx.add_user_message(f"Message {i}")

        # Should have max_length messages (smart trimming keeps most recent)
        assert len(ctx.messages) == 5
        # Should contain the last 5 messages
        assert ctx.messages[-1].content == "Message 9"

    def test_trim_preserves_system_message(self):
        """Test that trimming preserves the system message."""
        ctx = ConversationContext(max_length=5, system_prompt="System")
        for i in range(10):
            ctx.add_user_message(f"Message {i}")

        # First message should still be system
        assert ctx.messages[0].role == Role.SYSTEM
        assert ctx.messages[0].content == "System"
        # Total should be max_length
        assert len(ctx.messages) == 5

    def test_token_estimate(self):
        """Test token estimation."""
        ctx = ConversationContext()
        ctx.add_user_message("Hello world")  # ~6 chars / 4 = ~1-2 tokens + overhead

        estimate = ctx.token_estimate()
        assert estimate > 0  # Should have some tokens


class TestConversationPersistence:
    """Tests for conversation history persistence."""

    def test_save_history(self):
        """Test saving conversation history to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            history_file = Path(tmpdir) / "history.json"
            ctx = ConversationContext(
                system_prompt="Test system",
                history_file=history_file,
            )
            ctx.add_user_message("Test message")

            # Manually trigger save
            ctx._save_history()

            assert history_file.exists()
            data = json.loads(history_file.read_text())
            assert data["system_prompt"] == "Test system"
            assert len(data["messages"]) == 2  # system + user message

    def test_load_history(self):
        """Test loading conversation history from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            history_file = Path(tmpdir) / "history.json"

            # Create initial history
            ctx1 = ConversationContext(
                system_prompt="Loaded system",
                history_file=history_file,
            )
            ctx1.add_user_message("Loaded message")
            ctx1._save_history()

            # Load in new context
            ctx2 = ConversationContext(
                system_prompt="Loaded system",
                history_file=history_file,
            )

            assert len(ctx2.messages) == 2
            assert ctx2.messages[0].content == "Loaded system"
            assert ctx2.messages[1].content == "Loaded message"

    def test_load_nonexistent_history(self):
        """Test that loading nonexistent history file doesn't fail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            history_file = Path(tmpdir) / "nonexistent.json"
            ctx = ConversationContext(history_file=history_file)

            # Should not raise any errors
            assert ctx.messages == []

    def test_save_history_creates_parent_directories(self):
        """Test that save_history creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            history_file = Path(tmpdir) / "nested" / "dir" / "history.json"
            ctx = ConversationContext(history_file=history_file)
            ctx.add_user_message("Test")
            ctx._save_history()

            assert history_file.exists()

    def test_save_history_silently_fails_on_error(self):
        """Test that save_history doesn't raise exceptions on failure."""
        # Use an invalid path that should fail
        ctx = ConversationContext(history_file=Path("/invalid/path/history.json"))
        ctx.add_user_message("Test")

        # Should not raise any exception
        ctx._save_history()

    def test_load_history_silently_fails_on_error(self):
        """Test that load_history doesn't raise exceptions on failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_file = Path(tmpdir) / "bad.json"
            bad_file.write_text("not valid json")

            ctx = ConversationContext(history_file=bad_file)

            # Should not raise any exception
            # Context should just have empty or minimal messages
            assert isinstance(ctx.messages, list)

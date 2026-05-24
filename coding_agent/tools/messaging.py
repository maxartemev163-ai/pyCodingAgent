"""Messaging and task management tools for Slack, Telegram, and Jira."""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from .base import Tool, ToolResult


@dataclass
class TaskMessage:
    """Represents a task message from a messaging platform.

    Attributes:
        id: Unique identifier for the message/task.
        content: The message/task content.
        source: Source platform (slack, telegram, jira).
        channel: Channel/room/project where the message originated.
        author: Author of the message.
        timestamp: When the message was created.
        metadata: Additional platform-specific metadata.
    """

    id: str
    content: str
    source: str
    channel: str = ""
    author: str = ""
    timestamp: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "content": self.content,
            "source": self.source,
            "channel": self.channel,
            "author": self.author,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class MessagingClient(ABC):
    """Abstract base class for messaging clients."""

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Return the platform name."""
        pass

    @abstractmethod
    def connect(self) -> ToolResult:
        """Establish connection to the platform."""
        pass

    @abstractmethod
    def disconnect(self) -> ToolResult:
        """Disconnect from the platform."""
        pass

    @abstractmethod
    def fetch_messages(self, channel: str, limit: int = 10) -> ToolResult:
        """Fetch messages from a channel."""
        pass

    @abstractmethod
    def send_message(self, channel: str, content: str) -> ToolResult:
        """Send a message to a channel."""
        pass


class SlackClient(MessagingClient):
    """Slack API client for receiving and sending messages.

    Requires a Slack Bot Token with appropriate scopes.
    Environment variable SLACK_BOT_TOKEN or token parameter must be provided.
    """

    def __init__(self, token: Optional[str] = None, api_url: str = "https://slack.com/api") -> None:
        """Initialize Slack client.

        Args:
            token: Slack Bot Token (can also be set via SLACK_BOT_TOKEN env var).
            api_url: Slack API base URL.
        """
        self.token = token
        self.api_url = api_url
        self._connected = False
        self._client: Optional[httpx.Client] = None

    @property
    def platform_name(self) -> str:
        """Return platform name."""
        return "slack"

    def _get_headers(self) -> dict:
        """Get authorization headers."""
        return {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}

    def connect(self) -> ToolResult:
        """Test connection to Slack API."""
        try:
            token = self.token or os.environ.get("SLACK_BOT_TOKEN")
            if not token:
                return ToolResult(
                    success=False,
                    error="Slack token not provided. Set SLACK_BOT_TOKEN env var or pass token parameter.",
                )

            self.token = token
            self._client = httpx.Client(headers=self._get_headers(), timeout=30.0)

            # Test connection with auth.test endpoint
            response = self._client.get(f"{self.api_url}/auth.test")
            data = response.json()

            if data.get("ok"):
                self._connected = True
                return ToolResult(success=True, output=f"Connected to Slack as {data.get('user_id')}")
            return ToolResult(success=False, error=data.get("error", "Unknown Slack API error"))

        except httpx.HTTPError as e:
            return ToolResult(success=False, error=f"HTTP error connecting to Slack: {str(e)}")
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to connect to Slack: {str(e)}")

    def disconnect(self) -> ToolResult:
        """Disconnect from Slack."""
        try:
            if self._client:
                self._client.close()
                self._client = None
            self._connected = False
            return ToolResult(success=True, output="Disconnected from Slack")
        except Exception as e:
            return ToolResult(success=False, error=f"Error disconnecting: {str(e)}")

    def fetch_messages(self, channel: str, limit: int = 10) -> ToolResult:
        """Fetch messages from a Slack channel.

        Args:
            channel: Channel ID or name (e.g., 'C123456' or '#general').
            limit: Maximum number of messages to fetch.

        Returns:
            ToolResult with list of TaskMessage objects.
        """
        try:
            if not self._connected:
                return ToolResult(success=False, error="Not connected to Slack. Call connect() first.")

            # Convert channel name to ID if needed
            channel_id = channel
            if channel.startswith("#"):
                resolve_result = self._resolve_channel_name(channel)
                if not resolve_result.success:
                    return resolve_result
                channel_id = resolve_result.output

            response = self._client.get(
                f"{self.api_url}/conversations.history",
                params={"channel": channel_id, "limit": limit},
            )
            data = response.json()

            if not data.get("ok"):
                return ToolResult(success=False, error=data.get("error", "Failed to fetch messages"))

            messages = []
            for msg in data.get("messages", []):
                task_msg = TaskMessage(
                    id=msg.get("ts", ""),
                    content=msg.get("text", ""),
                    source="slack",
                    channel=channel,
                    author=msg.get("user", ""),
                    timestamp=msg.get("ts", ""),
                    metadata=msg,
                )
                messages.append(task_msg)

            output = json.dumps([m.to_dict() for m in messages], indent=2)
            return ToolResult(success=True, output=output)

        except httpx.HTTPError as e:
            return ToolResult(success=False, error=f"HTTP error fetching messages: {str(e)}")
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to fetch messages: {str(e)}")

    def _resolve_channel_name(self, channel_name: str) -> ToolResult:
        """Resolve channel name to channel ID."""
        try:
            name = channel_name.lstrip("#")
            response = self._client.get(
                f"{self.api_url}/conversations.list",
                params={"types": "public_channel,private_channel"},
            )
            data = response.json()

            if not data.get("ok"):
                return ToolResult(success=False, error=data.get("error", "Failed to list channels"))

            for ch in data.get("channels", []):
                if ch.get("name") == name:
                    return ToolResult(success=True, output=ch.get("id"))

            return ToolResult(success=False, error=f"Channel '{channel_name}' not found")

        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def send_message(self, channel: str, content: str) -> ToolResult:
        """Send a message to a Slack channel.

        Args:
            channel: Channel ID or name.
            content: Message content.

        Returns:
            ToolResult with send status.
        """
        try:
            if not self._connected:
                return ToolResult(success=False, error="Not connected to Slack. Call connect() first.")

            channel_id = channel
            if channel.startswith("#"):
                resolve_result = self._resolve_channel_name(channel)
                if not resolve_result.success:
                    return resolve_result
                channel_id = resolve_result.output

            response = self._client.post(
                f"{self.api_url}/chat.postMessage",
                json={"channel": channel_id, "text": content},
            )
            data = response.json()

            if data.get("ok"):
                return ToolResult(
                    success=True, output=f"Message sent successfully (ts: {data.get('ts')})"
                )
            return ToolResult(success=False, error=data.get("error", "Failed to send message"))

        except httpx.HTTPError as e:
            return ToolResult(success=False, error=f"HTTP error sending message: {str(e)}")
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to send message: {str(e)}")


class TelegramClient(MessagingClient):
    """Telegram Bot API client for receiving and sending messages.

    Requires a Telegram Bot Token.
    Environment variable TELEGRAM_BOT_TOKEN or token parameter must be provided.
    """

    def __init__(self, token: Optional[str] = None, api_url: str = "https://api.telegram.org") -> None:
        """Initialize Telegram client.

        Args:
            token: Telegram Bot Token.
            api_url: Telegram API base URL.
        """
        self.token = token
        self.api_url = api_url
        self._connected = False
        self._client: Optional[httpx.Client] = None

    @property
    def platform_name(self) -> str:
        """Return platform name."""
        return "telegram"

    def _get_api_url(self, method: str) -> str:
        """Get full API URL for a method."""
        return f"{self.api_url}/bot{self.token}/{method}"

    def connect(self) -> ToolResult:
        """Test connection to Telegram API."""
        try:
            token = self.token or os.environ.get("TELEGRAM_BOT_TOKEN")
            if not token:
                return ToolResult(
                    success=False,
                    error="Telegram token not provided. Set TELEGRAM_BOT_TOKEN env var or pass token parameter.",
                )

            self.token = token
            self._client = httpx.Client(timeout=30.0)

            # Test connection with getMe endpoint
            response = self._client.get(self._get_api_url("getMe"))
            data = response.json()

            if data.get("ok"):
                self._connected = True
                bot = data.get("result", {})
                return ToolResult(
                    success=True, output=f"Connected to Telegram as @{bot.get('username', 'bot')}"
                )
            return ToolResult(success=False, error=data.get("description", "Unknown Telegram API error"))

        except httpx.HTTPError as e:
            return ToolResult(success=False, error=f"HTTP error connecting to Telegram: {str(e)}")
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to connect to Telegram: {str(e)}")

    def disconnect(self) -> ToolResult:
        """Disconnect from Telegram."""
        try:
            if self._client:
                self._client.close()
                self._client = None
            self._connected = False
            return ToolResult(success=True, output="Disconnected from Telegram")
        except Exception as e:
            return ToolResult(success=False, error=f"Error disconnecting: {str(e)}")

    def fetch_messages(self, channel: str, limit: int = 10) -> ToolResult:
        """Fetch recent updates from Telegram (simulated as chat history).

        Note: Telegram Bot API doesn't provide direct message history access.
        This uses getUpdates to fetch recent messages.

        Args:
            channel: Chat ID or username (e.g., '@channelname' or '-100123456').
            limit: Maximum number of messages to fetch.

        Returns:
            ToolResult with list of TaskMessage objects.
        """
        try:
            if not self._connected:
                return ToolResult(
                    success=False, error="Not connected to Telegram. Call connect() first."
                )

            response = self._client.get(
                self._get_api_url("getUpdates"),
                params={"limit": limit, "timeout": 0},
            )
            data = response.json()

            if not data.get("ok"):
                return ToolResult(
                    success=False, error=data.get("description", "Failed to fetch updates")
                )

            messages = []
            for update in data.get("result", []):
                message = update.get("message", {})
                if not message:
                    continue

                chat = message.get("chat", {})
                chat_id = str(chat.get("id", ""))

                # Filter by channel if specified
                if channel and not channel.endswith(chat_id):
                    continue

                task_msg = TaskMessage(
                    id=str(message.get("message_id", "")),
                    content=message.get("text", ""),
                    source="telegram",
                    channel=f"@{chat.get('username', chat_id)}",
                    author=message.get("from", {}).get("username", "unknown"),
                    timestamp=str(message.get("date", "")),
                    metadata=message,
                )
                messages.append(task_msg)

            output = json.dumps([m.to_dict() for m in messages], indent=2)
            return ToolResult(success=True, output=output)

        except httpx.HTTPError as e:
            return ToolResult(success=False, error=f"HTTP error fetching messages: {str(e)}")
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to fetch messages: {str(e)}")

    def send_message(self, channel: str, content: str) -> ToolResult:
        """Send a message to a Telegram chat/channel.

        Args:
            channel: Chat ID or username (e.g., '@channelname').
            content: Message content.

        Returns:
            ToolResult with send status.
        """
        try:
            if not self._connected:
                return ToolResult(
                    success=False, error="Not connected to Telegram. Call connect() first."
                )

            chat_id = channel
            response = self._client.post(
                self._get_api_url("sendMessage"),
                json={"chat_id": chat_id, "text": content},
            )
            data = response.json()

            if data.get("ok"):
                msg = data.get("result", {})
                return ToolResult(
                    success=True, output=f"Message sent successfully (id: {msg.get('message_id')})"
                )
            return ToolResult(
                success=False, error=data.get("description", "Failed to send message")
            )

        except httpx.HTTPError as e:
            return ToolResult(success=False, error=f"HTTP error sending message: {str(e)}")
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to send message: {str(e)}")


class JiraClient(MessagingClient):
    """Jira API client for receiving and creating issues/tasks.

    Requires Jira instance URL, email, and API token.
    Environment variables JIRA_URL, JIRA_EMAIL, JIRA_TOKEN or parameters must be provided.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        email: Optional[str] = None,
        api_token: Optional[str] = None,
    ) -> None:
        """Initialize Jira client.

        Args:
            url: Jira instance URL (e.g., 'https://your-domain.atlassian.net').
            email: Jira account email.
            api_token: Jira API token.
        """
        self.url = url
        self.email = email
        self.api_token = api_token
        self._connected = False
        self._client: Optional[httpx.Client] = None

    @property
    def platform_name(self) -> str:
        """Return platform name."""
        return "jira"

    def _get_auth(self) -> tuple:
        """Get basic auth tuple."""
        return (self.email, self.api_token)

    def connect(self) -> ToolResult:
        """Test connection to Jira API."""
        try:
            url = self.url or os.environ.get("JIRA_URL")
            email = self.email or os.environ.get("JIRA_EMAIL")
            api_token = self.api_token or os.environ.get("JIRA_TOKEN")

            if not all([url, email, api_token]):
                return ToolResult(
                    success=False,
                    error="Jira credentials not provided. Set JIRA_URL, JIRA_EMAIL, JIRA_TOKEN env vars or pass parameters.",
                )

            self.url = url.rstrip("/")
            self.email = email
            self.api_token = api_token

            self._client = httpx.Client(auth=self._get_auth(), timeout=30.0)

            # Test connection with myPermissions endpoint
            response = self._client.get(f"{self.url}/rest/api/3/mypermissions")

            if response.status_code == 200:
                self._connected = True
                return ToolResult(success=True, output=f"Connected to Jira at {self.url}")
            return ToolResult(
                success=False, error=f"Failed to connect to Jira: {response.status_code}"
            )

        except httpx.HTTPError as e:
            return ToolResult(success=False, error=f"HTTP error connecting to Jira: {str(e)}")
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to connect to Jira: {str(e)}")

    def disconnect(self) -> ToolResult:
        """Disconnect from Jira."""
        try:
            if self._client:
                self._client.close()
                self._client = None
            self._connected = False
            return ToolResult(success=True, output="Disconnected from Jira")
        except Exception as e:
            return ToolResult(success=False, error=f"Error disconnecting: {str(e)}")

    def fetch_messages(self, channel: str, limit: int = 10) -> ToolResult:
        """Fetch issues/tasks from a Jira project or filter.

        Args:
            channel: Project key (e.g., 'PROJ') or JQL query.
            limit: Maximum number of issues to fetch.

        Returns:
            ToolResult with list of TaskMessage objects (issues).
        """
        try:
            if not self._connected:
                return ToolResult(success=False, error="Not connected to Jira. Call connect() first.")

            # Determine if channel is a project key or JQL
            if channel.upper() == channel and len(channel) <= 10:
                # Likely a project key
                jql = f"project = {channel} ORDER BY updated DESC"
            else:
                # Treat as JQL query
                jql = channel

            response = self._client.get(
                f"{self.url}/rest/api/3/search",
                params={"jql": jql, "maxResults": limit},
            )

            if response.status_code != 200:
                return ToolResult(
                    success=False, error=f"Failed to search issues: {response.status_code}"
                )

            data = response.json()
            issues = data.get("issues", [])

            messages = []
            for issue in issues:
                fields = issue.get("fields", {})
                task_msg = TaskMessage(
                    id=issue.get("key", ""),
                    content=f"{fields.get('summary', '')}: {fields.get('description', '')}",
                    source="jira",
                    channel=fields.get("project", {}).get("key", ""),
                    author=fields.get("creator", {}).get("displayName", ""),
                    timestamp=fields.get("created", ""),
                    metadata=fields,
                )
                messages.append(task_msg)

            output = json.dumps([m.to_dict() for m in messages], indent=2)
            return ToolResult(success=True, output=output)

        except httpx.HTTPError as e:
            return ToolResult(success=False, error=f"HTTP error fetching issues: {str(e)}")
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to fetch issues: {str(e)}")

    def send_message(self, channel: str, content: str) -> ToolResult:
        """Create a new Jira issue.

        Args:
            channel: Project key (e.g., 'PROJ').
            content: Issue summary/description (format: 'Summary|Description').

        Returns:
            ToolResult with creation status.
        """
        try:
            if not self._connected:
                return ToolResult(success=False, error="Not connected to Jira. Call connect() first.")

            # Parse content - format: "Summary|Description" or just "Summary"
            parts = content.split("|", 1)
            summary = parts[0].strip()
            description = parts[1].strip() if len(parts) > 1 else ""

            payload = {
                "fields": {
                    "project": {"key": channel},
                    "summary": summary,
                    "description": description,
                    "issuetype": {"name": "Task"},
                }
            }

            response = self._client.post(
                f"{self.url}/rest/api/3/issue",
                json=payload,
                headers={"Accept": "application/json", "Content-Type": "application/json"},
            )

            if response.status_code in [200, 201]:
                data = response.json()
                issue_key = data.get("key", "")
                return ToolResult(success=True, output=f"Issue created: {issue_key}")
            return ToolResult(
                success=False, error=f"Failed to create issue: {response.status_code} - {response.text}"
            )

        except httpx.HTTPError as e:
            return ToolResult(success=False, error=f"HTTP error creating issue: {str(e)}")
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to create issue: {str(e)}")


# Tool implementations following the existing pattern


class SlackReceiveTool(Tool):
    """Tool for receiving messages from Slack."""

    def __init__(self, token: Optional[str] = None) -> None:
        """Initialize the tool.

        Args:
            token: Slack Bot Token.
        """
        self._client = SlackClient(token=token)

    @property
    def name(self) -> str:
        """Return tool name."""
        return "slack_receive"

    @property
    def description(self) -> str:
        """Return tool description."""
        return (
            "Receive and analyze messages from a Slack channel. "
            "Useful for monitoring task assignments, feedback, or team communication. "
            "Requires SLACK_BOT_TOKEN environment variable or token parameter."
        )

    @property
    def schema(self) -> dict:
        """Return tool parameter schema."""
        return {
            "type": "object",
            "properties": {
                "channel": {
                    "type": "string",
                    "description": "Slack channel ID or name (e.g., '#general' or 'C123456')",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of messages to fetch (default: 10)",
                    "default": 10,
                },
            },
            "required": ["channel"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool.

        Args:
            **kwargs: Must contain 'channel', optionally 'limit'.

        Returns:
            ToolResult with messages.
        """
        channel = kwargs.get("channel")
        limit = kwargs.get("limit", 10)

        if not channel:
            return ToolResult(success=False, error="Missing required parameter: channel")

        # Connect first
        connect_result = self._client.connect()
        if not connect_result.success:
            return connect_result

        try:
            # Fetch messages
            return self._client.fetch_messages(channel, limit)
        finally:
            self._client.disconnect()


class SlackSendTool(Tool):
    """Tool for sending messages to Slack."""

    def __init__(self, token: Optional[str] = None) -> None:
        """Initialize the tool.

        Args:
            token: Slack Bot Token.
        """
        self._client = SlackClient(token=token)

    @property
    def name(self) -> str:
        """Return tool name."""
        return "slack_send"

    @property
    def description(self) -> str:
        """Return tool description."""
        return (
            "Send a message to a Slack channel. "
            "Useful for notifications, status updates, or task confirmations. "
            "Requires SLACK_BOT_TOKEN environment variable or token parameter."
        )

    @property
    def schema(self) -> dict:
        """Return tool parameter schema."""
        return {
            "type": "object",
            "properties": {
                "channel": {
                    "type": "string",
                    "description": "Slack channel ID or name (e.g., '#general' or 'C123456')",
                },
                "message": {
                    "type": "string",
                    "description": "Message content to send",
                },
            },
            "required": ["channel", "message"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool.

        Args:
            **kwargs: Must contain 'channel' and 'message'.

        Returns:
            ToolResult with send status.
        """
        channel = kwargs.get("channel")
        message = kwargs.get("message")

        if not channel:
            return ToolResult(success=False, error="Missing required parameter: channel")
        if not message:
            return ToolResult(success=False, error="Missing required parameter: message")

        # Connect first
        connect_result = self._client.connect()
        if not connect_result.success:
            return connect_result

        try:
            return self._client.send_message(channel, message)
        finally:
            self._client.disconnect()


class TelegramReceiveTool(Tool):
    """Tool for receiving messages from Telegram."""

    def __init__(self, token: Optional[str] = None) -> None:
        """Initialize the tool.

        Args:
            token: Telegram Bot Token.
        """
        self._client = TelegramClient(token=token)

    @property
    def name(self) -> str:
        """Return tool name."""
        return "telegram_receive"

    @property
    def description(self) -> str:
        """Return tool description."""
        return (
            "Receive and analyze messages from a Telegram chat or channel. "
            "Useful for monitoring task assignments or team communication. "
            "Requires TELEGRAM_BOT_TOKEN environment variable or token parameter."
        )

    @property
    def schema(self) -> dict:
        """Return tool parameter schema."""
        return {
            "type": "object",
            "properties": {
                "channel": {
                    "type": "string",
                    "description": "Telegram chat ID or username (e.g., '@channelname')",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of messages to fetch (default: 10)",
                    "default": 10,
                },
            },
            "required": ["channel"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool.

        Args:
            **kwargs: Must contain 'channel', optionally 'limit'.

        Returns:
            ToolResult with messages.
        """
        channel = kwargs.get("channel")
        limit = kwargs.get("limit", 10)

        if not channel:
            return ToolResult(success=False, error="Missing required parameter: channel")

        # Connect first
        connect_result = self._client.connect()
        if not connect_result.success:
            return connect_result

        try:
            return self._client.fetch_messages(channel, limit)
        finally:
            self._client.disconnect()


class TelegramSendTool(Tool):
    """Tool for sending messages to Telegram."""

    def __init__(self, token: Optional[str] = None) -> None:
        """Initialize the tool.

        Args:
            token: Telegram Bot Token.
        """
        self._client = TelegramClient(token=token)

    @property
    def name(self) -> str:
        """Return tool name."""
        return "telegram_send"

    @property
    def description(self) -> str:
        """Return tool description."""
        return (
            "Send a message to a Telegram chat or channel. "
            "Useful for notifications, status updates, or task confirmations. "
            "Requires TELEGRAM_BOT_TOKEN environment variable or token parameter."
        )

    @property
    def schema(self) -> dict:
        """Return tool parameter schema."""
        return {
            "type": "object",
            "properties": {
                "channel": {
                    "type": "string",
                    "description": "Telegram chat ID or username (e.g., '@channelname')",
                },
                "message": {
                    "type": "string",
                    "description": "Message content to send",
                },
            },
            "required": ["channel", "message"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool.

        Args:
            **kwargs: Must contain 'channel' and 'message'.

        Returns:
            ToolResult with send status.
        """
        channel = kwargs.get("channel")
        message = kwargs.get("message")

        if not channel:
            return ToolResult(success=False, error="Missing required parameter: channel")
        if not message:
            return ToolResult(success=False, error="Missing required parameter: message")

        # Connect first
        connect_result = self._client.connect()
        if not connect_result.success:
            return connect_result

        try:
            return self._client.send_message(channel, message)
        finally:
            self._client.disconnect()


class JiraReceiveTool(Tool):
    """Tool for receiving issues/tasks from Jira."""

    def __init__(
        self, url: Optional[str] = None, email: Optional[str] = None, api_token: Optional[str] = None
    ) -> None:
        """Initialize the tool.

        Args:
            url: Jira instance URL.
            email: Jira account email.
            api_token: Jira API token.
        """
        self._client = JiraClient(url=url, email=email, api_token=api_token)

    @property
    def name(self) -> str:
        """Return tool name."""
        return "jira_receive"

    @property
    def description(self) -> str:
        """Return tool description."""
        return (
            "Receive and analyze issues/tasks from a Jira project. "
            "Useful for monitoring assigned tasks, bugs, or feature requests. "
            "Requires JIRA_URL, JIRA_EMAIL, JIRA_TOKEN environment variables or parameters."
        )

    @property
    def schema(self) -> dict:
        """Return tool parameter schema."""
        return {
            "type": "object",
            "properties": {
                "project": {
                    "type": "string",
                    "description": "Jira project key (e.g., 'PROJ') or JQL query",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of issues to fetch (default: 10)",
                    "default": 10,
                },
            },
            "required": ["project"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool.

        Args:
            **kwargs: Must contain 'project', optionally 'limit'.

        Returns:
            ToolResult with issues.
        """
        project = kwargs.get("project")
        limit = kwargs.get("limit", 10)

        if not project:
            return ToolResult(success=False, error="Missing required parameter: project")

        # Connect first
        connect_result = self._client.connect()
        if not connect_result.success:
            return connect_result

        try:
            return self._client.fetch_messages(project, limit)
        finally:
            self._client.disconnect()


class JiraCreateTool(Tool):
    """Tool for creating issues/tasks in Jira."""

    def __init__(
        self, url: Optional[str] = None, email: Optional[str] = None, api_token: Optional[str] = None
    ) -> None:
        """Initialize the tool.

        Args:
            url: Jira instance URL.
            email: Jira account email.
            api_token: Jira API token.
        """
        self._client = JiraClient(url=url, email=email, api_token=api_token)

    @property
    def name(self) -> str:
        """Return tool name."""
        return "jira_create"

    @property
    def description(self) -> str:
        """Return tool description."""
        return (
            "Create a new issue/task in Jira. "
            "Content format: 'Summary|Description' or just 'Summary'. "
            "Requires JIRA_URL, JIRA_EMAIL, JIRA_TOKEN environment variables or parameters."
        )

    @property
    def schema(self) -> dict:
        """Return tool parameter schema."""
        return {
            "type": "object",
            "properties": {
                "project": {
                    "type": "string",
                    "description": "Jira project key (e.g., 'PROJ')",
                },
                "content": {
                    "type": "string",
                    "description": "Issue content in format 'Summary|Description'",
                },
            },
            "required": ["project", "content"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool.

        Args:
            **kwargs: Must contain 'project' and 'content'.

        Returns:
            ToolResult with creation status.
        """
        project = kwargs.get("project")
        content = kwargs.get("content")

        if not project:
            return ToolResult(success=False, error="Missing required parameter: project")
        if not content:
            return ToolResult(success=False, error="Missing required parameter: content")

        # Connect first
        connect_result = self._client.connect()
        if not connect_result.success:
            return connect_result

        try:
            return self._client.send_message(project, content)
        finally:
            self._client.disconnect()


class AnalyzeTasksTool(Tool):
    """Tool for analyzing tasks from multiple sources."""

    def __init__(self) -> None:
        """Initialize the analysis tool."""
        pass

    @property
    def name(self) -> str:
        """Return tool name."""
        return "analyze_tasks"

    @property
    def description(self) -> str:
        """Return tool description."""
        return (
            "Analyze task messages from Slack, Telegram, or Jira to extract action items, "
            "priorities, and assignees. Provides structured analysis of incoming tasks."
        )

    @property
    def schema(self) -> dict:
        """Return tool parameter schema."""
        return {
            "type": "object",
            "properties": {
                "messages_json": {
                    "type": "string",
                    "description": "JSON array of task messages to analyze",
                },
                "analysis_type": {
                    "type": "string",
                    "description": "Type of analysis: 'summary', 'extract_actions', 'prioritize'",
                    "enum": ["summary", "extract_actions", "prioritize"],
                    "default": "summary",
                },
            },
            "required": ["messages_json"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool.

        Args:
            **kwargs: Must contain 'messages_json', optionally 'analysis_type'.

        Returns:
            ToolResult with analysis results.
        """
        try:
            messages_json = kwargs.get("messages_json")
            analysis_type = kwargs.get("analysis_type", "summary")

            if not messages_json:
                return ToolResult(success=False, error="Missing required parameter: messages_json")

            messages = json.loads(messages_json)

            if not isinstance(messages, list):
                return ToolResult(success=False, error="messages_json must be a JSON array")

            analysis = self._analyze(messages, analysis_type)
            return ToolResult(success=True, output=json.dumps(analysis, indent=2))

        except json.JSONDecodeError as e:
            return ToolResult(success=False, error=f"Invalid JSON: {str(e)}")
        except Exception as e:
            return ToolResult(success=False, error=f"Analysis failed: {str(e)}")

    def _analyze(self, messages: list[dict], analysis_type: str) -> dict:
        """Perform analysis on messages."""
        result = {
            "total_messages": len(messages),
            "sources": {},
            "analysis_type": analysis_type,
        }

        # Count by source
        for msg in messages:
            source = msg.get("source", "unknown")
            result["sources"][source] = result["sources"].get(source, 0) + 1

        if analysis_type == "summary":
            result["summary"] = self._generate_summary(messages)
        elif analysis_type == "extract_actions":
            result["actions"] = self._extract_actions(messages)
        elif analysis_type == "prioritize":
            result["prioritized"] = self._prioritize_tasks(messages)

        return result

    def _generate_summary(self, messages: list[dict]) -> str:
        """Generate a summary of messages."""
        if not messages:
            return "No messages to summarize."

        authors = set(msg.get("author", "unknown") for msg in messages)
        channels = set(msg.get("channel", "unknown") for msg in messages)

        summary_parts = [
            f"Received {len(messages)} messages",
            f"From {len(authors)} unique author(s): {', '.join(authors)}",
            f"Across {len(channels)} channel(s): {', '.join(channels)}",
        ]

        return ". ".join(summary_parts)

    def _extract_actions(self, messages: list[dict]) -> list[dict]:
        """Extract action items from messages."""
        actions = []
        action_keywords = [
            "please",
            "need to",
            "should",
            "must",
            "action",
            "task",
            "do this",
            "assign",
            "review",
            "fix",
            "implement",
            "create",
            "update",
        ]

        for msg in messages:
            content = msg.get("content", "").lower()
            if any(keyword in content for keyword in action_keywords):
                actions.append(
                    {
                        "id": msg.get("id"),
                        "content": msg.get("content"),
                        "author": msg.get("author"),
                        "source": msg.get("source"),
                    }
                )

        return actions

    def _prioritize_tasks(self, messages: list[dict]) -> list[dict]:
        """Prioritize tasks based on keywords."""
        priority_high = ["urgent", "critical", "asap", "emergency", "high priority", "blocker"]
        priority_medium = ["important", "soon", "this week", "medium priority"]
        priority_low = ["when possible", "low priority", "nice to have", "backlog"]

        prioritized = {"high": [], "medium": [], "low": [], "unclassified": []}

        for msg in messages:
            content = msg.get("content", "").lower()
            classified = False

            for keyword in priority_high:
                if keyword in content:
                    prioritized["high"].append(msg)
                    classified = True
                    break

            if not classified:
                for keyword in priority_medium:
                    if keyword in content:
                        prioritized["medium"].append(msg)
                        classified = True
                        break

            if not classified:
                for keyword in priority_low:
                    if keyword in content:
                        prioritized["low"].append(msg)
                        classified = True
                        break

            if not classified:
                prioritized["unclassified"].append(msg)

        return {
            "high_priority_count": len(prioritized["high"]),
            "medium_priority_count": len(prioritized["medium"]),
            "low_priority_count": len(prioritized["low"]),
            "unclassified_count": len(prioritized["unclassified"]),
            "tasks": prioritized,
        }

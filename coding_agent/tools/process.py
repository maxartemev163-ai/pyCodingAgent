"""Process and service management tools for the coding agent."""

import os
import signal
import subprocess
from typing import Any

from .base import Tool, ToolResult


# Global process registry - shared across all tool instances
_running_processes: dict[str, subprocess.Popen] = {}


class ProcessTools:
    """Collection of process and service management tools.

    Provides capabilities for starting, stopping, and monitoring system processes.
    """

    def __init__(self, workspace_root: str = ".") -> None:
        """Initialize process tools.

        Args:
            workspace_root: Working directory for process operations.
        """
        self.workspace_root = workspace_root

    @staticmethod
    def get_registry() -> dict[str, subprocess.Popen]:
        """Get the global process registry."""
        return _running_processes


class StartProcessTool(Tool):
    """Tool for starting a background process or service."""

    def __init__(self, workspace_root: str = ".") -> None:
        """Initialize the start process tool.

        Args:
            workspace_root: Working directory for process execution.
        """
        self._process_tools = ProcessTools(workspace_root)

    @property
    def name(self) -> str:
        """Return tool name."""
        return "start_process"

    @property
    def description(self) -> str:
        """Return tool description."""
        return (
            "Start a background process or service. The process runs independently "
            "and can be managed using its process ID. Use for starting servers, "
            "daemons, or long-running tasks."
        )

    @property
    def schema(self) -> dict:
        """Return tool parameter schema."""
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Command to execute as a background process",
                },
                "name": {
                    "type": "string",
                    "description": "Unique name to identify this process for later management",
                },
            },
            "required": ["command", "name"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the start process operation.

        Args:
            **kwargs: Must contain 'command' and 'name' keys.

        Returns:
            ToolResult with process ID or error.
        """
        try:
            command = kwargs.get("command")
            name = kwargs.get("name")

            if not command:
                return ToolResult(success=False, error="Missing required parameter: command")
            if not name:
                return ToolResult(success=False, error="Missing required parameter: name")

            if name in _running_processes:
                return ToolResult(
                    success=False, error=f"Process '{name}' is already running"
                )

            process = subprocess.Popen(
                command,
                shell=True,
                cwd=self._process_tools.workspace_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )

            _running_processes[name] = process

            return ToolResult(
                success=True,
                output=f"Process '{name}' started with PID {process.pid}",
            )

        except Exception as e:
            return ToolResult(success=False, error=str(e))


class StopProcessTool(Tool):
    """Tool for stopping a running process."""

    def __init__(self) -> None:
        """Initialize the stop process tool."""
        pass

    @property
    def name(self) -> str:
        """Return tool name."""
        return "stop_process"

    @property
    def description(self) -> str:
        """Return tool description."""
        return (
            "Stop a running process by its name. Sends SIGTERM first, then SIGKILL "
            "if the process doesn't terminate gracefully."
        )

    @property
    def schema(self) -> dict:
        """Return tool parameter schema."""
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the process to stop",
                },
            },
            "required": ["name"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the stop process operation.

        Args:
            **kwargs: Must contain 'name' key.

        Returns:
            ToolResult with status or error.
        """
        try:
            name = kwargs.get("name")

            if not name:
                return ToolResult(success=False, error="Missing required parameter: name")

            if name not in _running_processes:
                return ToolResult(
                    success=False, error=f"Process '{name}' not found or not tracked"
                )

            process = _running_processes[name]

            if process.poll() is not None:
                del _running_processes[name]
                return ToolResult(
                    success=False, error=f"Process '{name}' already terminated"
                )

            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)

            del _running_processes[name]

            return ToolResult(success=True, output=f"Process '{name}' stopped successfully")

        except Exception as e:
            return ToolResult(success=False, error=str(e))


class ListProcessesTool(Tool):
    """Tool for listing running processes."""

    def __init__(self) -> None:
        """Initialize the list processes tool."""
        pass

    @property
    def name(self) -> str:
        """Return tool name."""
        return "list_processes"

    @property
    def description(self) -> str:
        """Return tool description."""
        return (
            "List all currently tracked running processes with their names and PIDs."
        )

    @property
    def schema(self) -> dict:
        """Return tool parameter schema."""
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the list processes operation.

        Args:
            **kwargs: No parameters required.

        Returns:
            ToolResult with list of processes.
        """
        try:
            processes = []
            for name, process in _running_processes.items():
                status = "running" if process.poll() is None else "terminated"
                processes.append(f"{name}: PID {process.pid} ({status})")

            output = "\n".join(processes) if processes else "(no processes running)"
            return ToolResult(success=True, output=output)

        except Exception as e:
            return ToolResult(success=False, error=str(e))


class GetProcessStatusTool(Tool):
    """Tool for getting status of a specific process."""

    def __init__(self) -> None:
        """Initialize the get process status tool."""
        pass

    @property
    def name(self) -> str:
        """Return tool name."""
        return "get_process_status"

    @property
    def description(self) -> str:
        """Return tool description."""
        return "Get the status of a specific process by name, including PID and whether it's still running."

    @property
    def schema(self) -> dict:
        """Return tool parameter schema."""
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the process to check",
                },
            },
            "required": ["name"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the get process status operation.

        Args:
            **kwargs: Must contain 'name' key.

        Returns:
            ToolResult with process status or error.
        """
        try:
            name = kwargs.get("name")

            if not name:
                return ToolResult(success=False, error="Missing required parameter: name")

            if name not in _running_processes:
                return ToolResult(
                    success=False, error=f"Process '{name}' not found or not tracked"
                )

            process = _running_processes[name]
            is_running = process.poll() is None

            if is_running:
                return ToolResult(
                    success=True,
                    output=f"Process '{name}' (PID {process.pid}) is running",
                )
            else:
                del _running_processes[name]
                return ToolResult(
                    success=True,
                    output=f"Process '{name}' (PID {process.pid}) has terminated",
                )

        except Exception as e:
            return ToolResult(success=False, error=str(e))


class KillProcessByIdTool(Tool):
    """Tool for killing a process by system PID."""

    def __init__(self) -> None:
        """Initialize the kill process by ID tool."""
        pass

    @property
    def name(self) -> str:
        """Return tool name."""
        return "kill_process_by_id"

    @property
    def description(self) -> str:
        """Return tool description."""
        return (
            "Kill a system process by its PID. Use with caution as this affects "
            "system-wide processes, not just those started by the agent."
        )

    @property
    def schema(self) -> dict:
        """Return tool parameter schema."""
        return {
            "type": "object",
            "properties": {
                "pid": {
                    "type": "integer",
                    "description": "Process ID to kill",
                },
                "signal": {
                    "type": "string",
                    "description": "Signal to send (default: TERM). Options: TERM, KILL, INT, HUP",
                },
            },
            "required": ["pid"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the kill process operation.

        Args:
            **kwargs: Must contain 'pid' key, optionally 'signal'.

        Returns:
            ToolResult with status or error.
        """
        try:
            pid = kwargs.get("pid")
            sig_name = kwargs.get("signal", "TERM")

            if pid is None:
                return ToolResult(success=False, error="Missing required parameter: pid")

            signal_map = {
                "TERM": signal.SIGTERM,
                "KILL": signal.SIGKILL,
                "INT": signal.SIGINT,
                "HUP": signal.SIGHUP,
            }

            if sig_name not in signal_map:
                return ToolResult(
                    success=False,
                    error=f"Unknown signal '{sig_name}'. Valid options: {list(signal_map.keys())}",
                )

            sig = signal_map[sig_name]

            try:
                os.kill(pid, sig)
                return ToolResult(
                    success=True,
                    output=f"Sent {sig_name} signal to process {pid}",
                )
            except ProcessLookupError:
                return ToolResult(success=False, error=f"Process {pid} not found")
            except PermissionError:
                return ToolResult(
                    success=False, error=f"Permission denied to kill process {pid}"
                )

        except Exception as e:
            return ToolResult(success=False, error=str(e))

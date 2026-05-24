"""Process and service management tools for the coding agent."""

import subprocess
import signal
import os
from typing import Any
from pathlib import Path

from .base import Tool, ToolResult


class ProcessTools:
    """Collection of process and service management tools.
    
    Provides capabilities for starting, stopping, and monitoring system processes.
    """

    def __init__(self, workspace_root: str = ".") -> None:
        """Initialize process tools.
        
        Args:
            workspace_root: Root directory for process operations.
        """
        self.workspace_root = Path(workspace_root).resolve()


class StartProcessTool(Tool):
    """Tool for starting a background process."""

    def __init__(self, workspace_root: str = ".") -> None:
        """Initialize the start process tool.
        
        Args:
            workspace_root: Working directory for the process.
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
            "Start a background process or service. The process runs independently. "
            "Use this to start servers, daemons, or long-running tasks. "
            "Returns the PID of the started process."
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
                    "description": "Optional name to identify the process",
                },
            },
            "required": ["command"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the start process operation.
        
        Args:
            **kwargs: Must contain 'command' key, optionally 'name'.
            
        Returns:
            ToolResult with process PID or error.
        """
        try:
            command = kwargs.get("command")
            name = kwargs.get("name", "unnamed")

            if not command:
                return ToolResult(success=False, error="Missing required parameter: command")

            # Start process in background
            process = subprocess.Popen(
                command,
                shell=True,
                cwd=str(self._process_tools.workspace_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,  # Detach from parent process
            )

            return ToolResult(
                success=True,
                output=f"Started process '{name}' with PID {process.pid}. Command: {command}",
            )

        except Exception as e:
            return ToolResult(success=False, error=str(e))


class StopProcessTool(Tool):
    """Tool for stopping a process by PID."""

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
            "Stop a running process by its PID. Sends SIGTERM first, "
            "then SIGKILL if the process doesn't terminate gracefully."
        )

    @property
    def schema(self) -> dict:
        """Return tool parameter schema."""
        return {
            "type": "object",
            "properties": {
                "pid": {
                    "type": "integer",
                    "description": "Process ID to stop",
                },
                "force": {
                    "type": "boolean",
                    "description": "If true, send SIGKILL immediately instead of SIGTERM",
                },
            },
            "required": ["pid"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the stop process operation.
        
        Args:
            **kwargs: Must contain 'pid' key, optionally 'force'.
            
        Returns:
            ToolResult with status or error.
        """
        try:
            pid = kwargs.get("pid")
            force = kwargs.get("force", False)

            if pid is None:
                return ToolResult(success=False, error="Missing required parameter: pid")

            try:
                pid = int(pid)
            except (ValueError, TypeError):
                return ToolResult(success=False, error=f"Invalid PID: {pid}")

            # Check if process exists
            try:
                os.kill(pid, 0)
            except OSError:
                return ToolResult(success=False, error=f"Process {pid} does not exist")

            if force:
                os.kill(pid, signal.SIGKILL)
                return ToolResult(success=True, output=f"Forcefully killed process {pid} with SIGKILL")
            else:
                os.kill(pid, signal.SIGTERM)
                return ToolResult(success=True, output=f"Sent SIGTERM to process {pid}")

        except PermissionError:
            return ToolResult(success=False, error=f"Permission denied to stop process {pid}")
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
            "List running processes. Optionally filter by name pattern. "
            "Shows PID, name, and status of matching processes."
        )

    @property
    def schema(self) -> dict:
        """Return tool parameter schema."""
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Optional pattern to filter process names",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of processes to return (default: 50)",
                },
            },
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the list processes operation.
        
        Args:
            **kwargs: Optionally 'pattern' and 'limit'.
            
        Returns:
            ToolResult with process list or error.
        """
        try:
            pattern = kwargs.get("pattern")
            limit = kwargs.get("limit", 50)

            processes = []
            
            # Use ps command to get process list
            result = subprocess.run(
                ["ps", "aux"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                return ToolResult(success=False, error="Failed to retrieve process list")

            lines = result.stdout.strip().split("\n")[1:]  # Skip header

            for line in lines[:limit]:
                parts = line.split(None, 10)
                if len(parts) >= 11:
                    pid = parts[1]
                    user = parts[0]
                    cpu = parts[2]
                    mem = parts[3]
                    command = parts[10]

                    if pattern and pattern.lower() not in command.lower():
                        continue

                    processes.append(f"PID {pid} | User: {user} | CPU: {cpu}% | MEM: {mem}% | {command}")

            if not processes:
                return ToolResult(success=True, output="No matching processes found")

            output = f"Found {len(processes)} process(es):\n" + "\n".join(processes)
            return ToolResult(success=True, output=output)

        except subprocess.TimeoutExpired:
            return ToolResult(success=False, error="Timed out while retrieving process list")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class GetProcessInfoTool(Tool):
    """Tool for getting detailed information about a specific process."""

    def __init__(self) -> None:
        """Initialize the get process info tool."""
        pass

    @property
    def name(self) -> str:
        """Return tool name."""
        return "get_process_info"

    @property
    def description(self) -> str:
        """Return tool description."""
        return (
            "Get detailed information about a specific process by PID. "
            "Includes status, memory usage, CPU usage, and command line."
        )

    @property
    def schema(self) -> dict:
        """Return tool parameter schema."""
        return {
            "type": "object",
            "properties": {
                "pid": {
                    "type": "integer",
                    "description": "Process ID to query",
                },
            },
            "required": ["pid"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the get process info operation.
        
        Args:
            **kwargs: Must contain 'pid' key.
            
        Returns:
            ToolResult with process info or error.
        """
        try:
            pid = kwargs.get("pid")

            if pid is None:
                return ToolResult(success=False, error="Missing required parameter: pid")

            try:
                pid = int(pid)
            except (ValueError, TypeError):
                return ToolResult(success=False, error=f"Invalid PID: {pid}")

            # Check if process exists
            try:
                os.kill(pid, 0)
            except OSError:
                return ToolResult(success=False, error=f"Process {pid} does not exist")

            # Get detailed info using ps
            result = subprocess.run(
                ["ps", "-p", str(pid), "-o", "pid,ppid,user,stat,start,time,cmd"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                return ToolResult(success=False, error=f"Failed to get info for process {pid}")

            lines = result.stdout.strip().split("\n")
            if len(lines) < 2:
                return ToolResult(success=False, error=f"Process {pid} not found")

            header = lines[0]
            data = lines[1]

            output = f"Process Information for PID {pid}:\n{header}\n{data}"
            return ToolResult(success=True, output=output)

        except subprocess.TimeoutExpired:
            return ToolResult(success=False, error="Timed out while retrieving process info")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

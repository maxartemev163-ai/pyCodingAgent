"""Context preparation for coding agent sessions.

This module provides functionality to prepare contextual information
before starting a session with the LLM, including:
- List of all files including subdirectories
- OS and datetime information
- pip freeze output and requirements.txt content
- Python coding rules and best practices
"""

import logging
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SessionContext:
    """Container for session context information.

    Attributes:
        file_list: List of all files in the workspace.
        os_info: Operating system information.
        datetime_info: Current date and time.
        pip_freeze: Output of pip freeze command.
        requirements: Content of requirements.txt file.
        python_rules: Python coding rules and best practices.
    """

    file_list: list[str]
    os_info: str
    datetime_info: str
    pip_freeze: str
    requirements: str
    python_rules: str

    def to_system_prompt(self) -> str:
        """Convert session context to a system prompt section.

        Returns:
            Formatted string containing all context information.
        """
        sections = []

        # Environment Information
        sections.append("ENVIRONMENT INFORMATION:")
        sections.append(f"Operating System: {self.os_info}")
        sections.append(f"Current DateTime: {self.datetime_info}")
        sections.append("")

        # Project Files
        sections.append("PROJECT FILES:")
        if self.file_list:
            sections.append("Files in workspace:")
            for file_path in self.file_list:
                sections.append(f"  - {file_path}")
        else:
            sections.append("  (No files found)")
        sections.append("")

        # Dependencies
        sections.append("DEPENDENCIES:")
        sections.append("Installed packages (pip freeze):")
        if self.pip_freeze:
            for line in self.pip_freeze.splitlines():
                sections.append(f"  {line}")
        else:
            sections.append("  (No packages listed)")
        sections.append("")

        sections.append("Project requirements.txt:")
        if self.requirements:
            for line in self.requirements.splitlines():
                sections.append(f"  {line}")
        else:
            sections.append("  (No requirements.txt found)")
        sections.append("")

        # Python Coding Rules
        sections.append("PYTHON CODING RULES:")
        sections.append(self.python_rules)
        sections.append("")

        return "\n".join(sections)


def get_file_list(root_dir: str = ".", max_depth: int = 2, max_files: int = 50) -> list[str]:
    """Get a list of all files in the directory tree.

    Args:
        root_dir: Root directory to search from.
        max_depth: Maximum depth to traverse (prevents deep recursion).
        max_files: Maximum number of files to return (limits context size).

    Returns:
        List of file paths relative to root_dir.
    """
    files = []
    root_path = Path(root_dir).resolve()

    for dirpath, _, filenames in os.walk(root_path):
        # Skip hidden directories and common non-essential directories
        dir_path = Path(dirpath)
        relative_dir_parts = dir_path.relative_to(root_path).parts
        
        # Check depth limit
        if len(relative_dir_parts) > max_depth:
            continue
            
        if any(part.startswith(".") for part in relative_dir_parts):
            continue
        if "__pycache__" in dir_path.parts:
            continue
        if ".git" in dir_path.parts:
            continue

        for filename in filenames:
            # Skip hidden files and compiled Python files
            if filename.startswith("."):
                continue
            if filename.endswith((".pyc", ".pyo")):
                continue

            file_path = Path(dirpath) / filename
            relative_path = file_path.relative_to(root_path)
            files.append(str(relative_path))
            
            # Early exit if we've reached max files
            if len(files) >= max_files:
                logger.warning(f"File list truncated at {max_files} files. Use more specific queries for deeper files.")
                return sorted(files)

    return sorted(files)


def get_os_info() -> str:
    """Get operating system information.

    Returns:
        String containing OS information.
    """
    info = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }

    return (
        f"{info['system']} {info['release']} ({info['version']})\n"
        f"Machine: {info['machine']}, Processor: {info['processor']}\n"
        f"Python Version: {info['python_version']}"
    )


def get_datetime_info() -> str:
    """Get current date and time information.

    Returns:
        Formatted datetime string.
    """
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")


def get_pip_freeze() -> str:
    """Get the output of pip freeze command.

    Returns:
        String containing pip freeze output.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error getting pip freeze: {e}"
    except subprocess.TimeoutExpired:
        return "Error: pip freeze command timed out"
    except Exception as e:
        return f"Error: {e}"


def read_requirements(requirements_file: str = "requirements.txt") -> str:
    """Read the contents of requirements.txt file.

    Args:
        requirements_file: Path to requirements file.

    Returns:
        Content of requirements.txt or empty string if not found.
    """
    req_path = Path(requirements_file)

    if not req_path.exists():
        return ""

    try:
        return req_path.read_text().strip()
    except Exception as e:
        return f"Error reading requirements.txt: {e}"


def get_python_coding_rules() -> str:
    """Get Python coding rules and best practices (concise version for small LLMs).

    Returns:
        String containing essential Python coding guidelines.
    """
    return """1. PEP 8 Style: 4 spaces, snake_case variables/functions, PascalCase classes, 79 char lines
2. Type Hints: Always annotate function parameters and return values
3. Error Handling: Use specific exceptions, never bare except; log errors properly
4. Security: Never hardcode secrets; use environment variables for config
5. Testing: Write unit tests for critical functions using pytest
6. Code Quality: Keep functions small and focused; avoid magic numbers
7. Documentation: Use docstrings (triple quotes); explain 'why' not 'what'
8. Imports: One per line at top; standard lib first, then third-party, then local"""


def prepare_session_context(
    workspace_dir: str = ".",
    requirements_file: str = "requirements.txt",
    include_files: bool = True,
    include_pip_freeze: bool = True,
    max_depth: int = 2,
    max_files: int = 50,
) -> SessionContext:
    """Prepare complete session context for the coding agent.

    This function gathers all necessary context information before
    starting a session with the LLM.

    Args:
        workspace_dir: Root directory of the workspace.
        requirements_file: Path to requirements.txt file.
        include_files: Whether to include file list (set False to reduce context).
        include_pip_freeze: Whether to include pip freeze output (set False for non-Python tasks).
        max_depth: Maximum directory depth for file listing.
        max_files: Maximum number of files to include.

    Returns:
        SessionContext object containing all prepared context.
    """
    return SessionContext(
        file_list=get_file_list(workspace_dir, max_depth=max_depth, max_files=max_files) if include_files else [],
        os_info=get_os_info(),
        datetime_info=get_datetime_info(),
        pip_freeze=get_pip_freeze() if include_pip_freeze else "",
        requirements=read_requirements(requirements_file) if include_pip_freeze else "",
        python_rules=get_python_coding_rules(),
    )

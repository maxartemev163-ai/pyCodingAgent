"""Settings configuration for the coding agent."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Settings:
    """Application settings for the coding agent.

    Attributes:
        workspace_dir: Root directory for the project workspace.
        max_iterations: Maximum number of iterations for agent loops.
        timeout_seconds: Timeout for individual tool executions.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        enable_history: Whether to persist conversation history.
        history_dir: Directory for storing conversation history.
    """

    workspace_dir: str = "."
    max_iterations: int = 15  # Reduced for small LLMs to prevent infinite loops
    timeout_seconds: int = 300
    log_level: str = "INFO"
    enable_history: bool = True
    history_dir: str = ".agent_history"
    max_context_length: int = 128000
    temperature: float = 0.2  # Lower temperature for more deterministic output (small LLMs)
    top_p: float = 0.9

    def __post_init__(self) -> None:
        """Validate settings after initialization."""
        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level not in valid_log_levels:
            raise ValueError(f"log_level must be one of {valid_log_levels}")

        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")

        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

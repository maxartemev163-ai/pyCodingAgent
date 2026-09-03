"""Tests for the configuration classes."""

import pytest

from coding_agent.config.model_config import ModelConfig
from coding_agent.config.settings import Settings


class TestModelConfig:
    """Tests for the ModelConfig dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = ModelConfig()

        assert config.base_url == "http://localhost:11434/v1"
        assert config.api_key == "ollama"
        assert config.model_name == "qwen2.5-coder:7b"
        assert config.max_tokens == 8128
        assert config.timeout == 10 * 60
        assert config.retry_count == 3
        assert config.stream is True

    def test_custom_values(self):
        """Test creating config with custom values."""
        config = ModelConfig(
            base_url="http://custom:8080/v1",
            api_key="my-key",
            model_name="gpt-4",
            max_tokens=2048,
            timeout=60,
            retry_count=5,
        )

        assert config.base_url == "http://custom:8080/v1"
        assert config.api_key == "my-key"
        assert config.model_name == "gpt-4"
        assert config.max_tokens == 2048
        assert config.timeout == 60
        assert config.retry_count == 5

    def test_empty_base_url_raises_error(self):
        """Test that empty base_url raises ValueError."""
        with pytest.raises(ValueError, match="base_url cannot be empty"):
            ModelConfig(base_url="")

    def test_empty_model_name_raises_error(self):
        """Test that empty model_name raises ValueError."""
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            ModelConfig(model_name="")

    def test_negative_max_tokens_raises_error(self):
        """Test that non-positive max_tokens raises ValueError."""
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            ModelConfig(max_tokens=0)
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            ModelConfig(max_tokens=-1)

    def test_negative_timeout_raises_error(self):
        """Test that non-positive timeout raises ValueError."""
        with pytest.raises(ValueError, match="timeout must be positive"):
            ModelConfig(timeout=0)
        with pytest.raises(ValueError, match="timeout must be positive"):
            ModelConfig(timeout=-10)

    def test_negative_retry_count_raises_error(self):
        """Test that negative retry_count raises ValueError."""
        with pytest.raises(ValueError, match="retry_count cannot be negative"):
            ModelConfig(retry_count=-1)

    def test_zero_retry_count_allowed(self):
        """Test that zero retry_count is allowed."""
        config = ModelConfig(retry_count=0)
        assert config.retry_count == 0


class TestSettings:
    """Tests for the Settings dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        settings = Settings()

        assert settings.workspace_dir == "."
        assert settings.max_iterations == 15
        assert settings.timeout_seconds == 300
        assert settings.log_level == "INFO"
        assert settings.enable_history is True
        assert settings.history_dir == ".agent_history"
        assert settings.max_context_length == 128000
        assert settings.temperature == 0.2
        assert settings.top_p == 0.9

    def test_custom_values(self):
        """Test creating settings with custom values."""
        settings = Settings(
            workspace_dir="/workspace",
            max_iterations=100,
            timeout_seconds=600,
            log_level="DEBUG",
            enable_history=False,
            history_dir="/tmp/history",
            max_context_length=64000,
            temperature=0.5,
            top_p=0.9,
        )

        assert settings.workspace_dir == "/workspace"
        assert settings.max_iterations == 100
        assert settings.timeout_seconds == 600
        assert settings.log_level == "DEBUG"
        assert settings.enable_history is False
        assert settings.history_dir == "/tmp/history"
        assert settings.max_context_length == 64000
        assert settings.temperature == 0.5
        assert settings.top_p == 0.9

    def test_invalid_log_level_raises_error(self):
        """Test that invalid log_level raises ValueError."""
        with pytest.raises(ValueError, match="log_level must be one of"):
            Settings(log_level="INVALID")

    def test_valid_log_levels(self):
        """Test that all valid log levels are accepted."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        for level in valid_levels:
            settings = Settings(log_level=level)
            assert settings.log_level == level

    def test_zero_max_iterations_raises_error(self):
        """Test that non-positive max_iterations raises ValueError."""
        with pytest.raises(ValueError, match="max_iterations must be positive"):
            Settings(max_iterations=0)
        with pytest.raises(ValueError, match="max_iterations must be positive"):
            Settings(max_iterations=-5)

    def test_zero_timeout_seconds_raises_error(self):
        """Test that non-positive timeout_seconds raises ValueError."""
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            Settings(timeout_seconds=0)
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            Settings(timeout_seconds=-100)

    def test_case_insensitive_log_level_validation(self):
        """Test that log level validation is case-sensitive (requires uppercase)."""
        # Lowercase should fail since validation checks against uppercase set
        with pytest.raises(ValueError):
            Settings(log_level="debug")

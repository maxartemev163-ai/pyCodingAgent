# LangChain Integration

This project has been refactored to use LangChain for LLM interactions, providing better optimization and comprehensive logging capabilities.

## Features

### LangChain Client

The new `LangChainClient` class provides:

1. **Comprehensive Logging**: All LLM requests and responses are automatically logged to JSON files
2. **Optimized API Interactions**: Uses LangChain's optimized ChatOpenAI client
3. **Message Conversion**: Seamless conversion between internal message format and LangChain messages
4. **Tool Call Support**: Native support for tool calls through LangChain's bind_tools

### Automatic Log Directory Creation

The logs directory is automatically created when the agent initializes:
- Default location: `<workspace>/logs/`
- Custom location can be specified via `logs_dir` parameter

### Log File Format

Each LLM interaction is saved as a JSON file with the following structure:

```json
{
  "timestamp": "2024-01-15T10:30:45.123456",
  "model": "qwen2.5-coder:7b",
  "request": {
    "messages": [...],
    "tools": [...],
    "parameters": {
      "temperature": 0.2,
      "top_p": 0.9,
      "max_tokens": 8128
    }
  },
  "response": {
    "content": "...",
    "tool_calls": [...],
    "metadata": {...}
  },
  "duration_seconds": 1.23
}
```

## Usage

### Basic Usage with LangChain (Default)

```python
from coding_agent.core.agent import CodingAgent
from coding_agent.config import ModelConfig

# Create agent with LangChain client (default)
agent = CodingAgent(
    model_config=ModelConfig(),
    client_type="langchain",  # Default
)

# All LLM interactions will be logged to <workspace>/logs/
response = agent.run("Create a hello world Python script")
agent.close()
```

### Custom Logs Directory

```python
from pathlib import Path

agent = CodingAgent(
    model_config=ModelConfig(),
    client_type="langchain",
    logs_dir="/path/to/custom/logs",  # Custom logs directory
)
```

### Using Native Client (Backward Compatibility)

```python
# Use the original httpx-based client without LangChain
agent = CodingAgent(
    model_config=ModelConfig(),
    client_type="native",  # Use native client
)
```

## Configuration

### Environment Variables

The LangChain client respects the same environment variables as the native client:

- `LLM_BASE_URL`: Base URL for the LLM API (default: `http://localhost:11434/v1`)
- `LLM_API_KEY`: API key for authentication (default: `ollama`)
- `LLM_MODEL`: Model name to use (default: `qwen2.5-coder:7b`)

### Model Configuration

```python
from coding_agent.config import ModelConfig

config = ModelConfig(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    model_name="qwen2.5-coder:7b",
    max_tokens=8128,
    timeout=600,
    temperature=0.2,
    top_p=0.9,
)

agent = CodingAgent(model_config=config)
```

## Benefits of LangChain Integration

1. **Better Optimization**: LangChain provides optimized request handling and caching
2. **Standardized Interface**: Consistent API across different LLM providers
3. **Enhanced Debugging**: Comprehensive logging of all interactions
4. **Future Extensibility**: Easy to add new features like memory, chains, and agents
5. **Community Support**: Leverages the large LangChain ecosystem

## Migration Guide

If you have existing code using the native client:

```python
# Old code
from coding_agent.llm import LLMClient
client = LLMClient(config)

# New code with LangChain
from coding_agent.llm import LangChainClient
client = LangChainClient(config)  # Automatically creates logs directory
```

Or simply use the agent with default settings:

```python
from coding_agent.core.agent import CodingAgent

# Defaults to LangChain client with logging enabled
agent = CodingAgent()
```

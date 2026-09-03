# Coding Agent

A modular and scale coding agent architecture designed for on-device exec w/ local LLM models.

## Features

- **OpenAI-Compatible API**: Works w/ local models like Qwen2.5-Coder via Ollama or any OpenAI-compatible endpoint
- **Modular Tool sys**: Easy to extend w/ custom tools
- **ReAct Pattern**: Reasoning Acting loop for complex task completion
- **Conversation History**: persist context across sessions
- **Google Style Guide**: Clean, well-documented code follow best practices
- **KISS DRY**: Simple, maintainable architecture
- **Desktop GUI**: Native PyQt5 desktop app for interactive coding (like Claude Code Desktop)

## Architecture

```
coding_agent/
 config/ # config classes (Settings, ModelConfig)
 core/ # Core agent logic (CodingAgent, ConversationContext)
 llm/ # LLM client (OpenAI-compatible)
 tools/ # Tool definitions and implementations
 utils/ # Utility funcs
 exs/ # Usage exs
```

## install

```
# Install deps
pip install httpx

# For GUI (optional)
pip install PyQt5

# For dev
pip install -r reqs-dev.txt
```

## Quick Start

### 1. Setup Local LLM

Install [Ollama](https://ollama.ai/) and pull a coding model:

```
ollama pull qwen2.5-coder:7b
```

### 2. Basic Usage

```
from coding_agent.config import ModelConfig, Settings
from coding_agent.core import CodingAgent
from coding_agent.tools import ReadFileTool, WriteFileTool, ListDirTool
from coding_agent.utils import setup_logging

setup_logging(level"INFO")

settings Settings(workspace_dir".")
model_config ModelConfig(
 base_url"http://localhost:11434/v1",
 model_name"qwen2.5-coder:7b",
)

w/ CodingAgent(settingssettings, model_configmodel_config) as agent:
 # Register tools
 agent.register_tool(ReadFileTool())
 agent.register_tool(WriteFileTool())
 agent.register_tool(ListDirTool())

 # Run agent
 resp agent.run("List all Python files in the curr dir")
 print(resp)
```

### 3. Run ex

```
python -m coding_agent.exs.basic_usage
```

### 4. Launch GUI (Desktop App)

For a desktop app experience similar to Claude Code Desktop:

```
# Install PyQt5 if not already installed
pip install PyQt5

# Launch the GUI
python -m coding_agent.gui
```

The GUI provides:
- Interactive chat interface with the coding agent
- File explorer panel showing your workspace
- Quick command buttons for common actions (Plan, Scan, Clear, Retry)
- Settings dialog to configure LLM endpoint and workspace
- Menu bar with tools and options
- Status bar showing agent state

## Creating Custom Tools

```
from typing import Any
from coding_agent.tools import Tool, ToolResult

class MyCustomTool(Tool):
 @property
 def name(self) - str:
 return "my_tool"

 @property
 def desc(self) - str:
 return "desc of what this tool does"

 @property
 def schema(self) - dict:
 return {
 "type": "object",
 "props": {
 "param1": {"type": "string", "desc": "Parameter desc"},
 },
 "req": ["param1"],
 }

 def run(self, **kwargs: Any) - ToolResult:
 param1 kwargs.get("param1")
 # Your impl here
 return ToolResult(successTrue, output"Result")
```

## Available Tools

| Tool | desc |
| `read_file` | Read contents of a file |
| `write_file` | Write content to a file |
| `list_dir` | List dir contents |
| `search_files` | Search files by glob pattern |
| `run_command` | run shell commands |
| `run_python` | run Python code snippets |
| `git_diff` | Check git diff against main/master branch |
| `git_commit` | mk a git commit w/ specified msg |
| `git_push` | Push git commits to remote repo |
| `git_pull_request` | mk pull reqs on GitHub or GitLab |

## config

### Settings

```
Settings(
 workspace_dir".", # Root dir for ops
 max_iterations50, # max agent loop iterations
 timeout_seconds300, # Tool exec timeout
 log_level"INFO", # Logging level
 enable_historyTrue, # Persist conversation history
 history_dir".agent_history",# History storage dir
 max_context_length128000, # Max context tokens
 temperature0.7, # LLM temperature
 top_p0.95, # LLM top_p
)
```

### ModelConfig

```
ModelConfig(
 base_url"http://localhost:11434/v1", # API endpoint
 api_key"ollama", # API key (dummy for local)
 model_name"qwen2.5-coder:7b", # Model name
 max_tokens4096, # Max resp tokens
 timeout120, # req timeout
 retry_count3, # Retry attempts
 streamTrue, // Stream resps
)
```

## Design Principles

### KISS (Keep It Simple, Stupid)
- Each tool has a single, clear responsibility
- Minimal abstraction layers
- Straightforward control flow

### DRY (Don't Repeat Yourself)
- Shared utils in common mods
- Centralized tool registry
- Reusable msg and config classes

### Google Style Guide
- complete docstrings
- Type hints throughout
- Clear naming conventions

## License

MIT License

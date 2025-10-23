# Open Agent Mode

An open-source implementation of ChatGPT's agent mode with tool use, code execution, and file operations.

## Features

- 🤖 **Multi-LLM Support**: Works with OpenAI, Anthropic, and local models (via Ollama/LlamaCPP)
- 🛠️ **Extensible Tool System**: Built-in tools for file operations, code execution, web search, and more
- 🔄 **Streaming Responses**: Real-time streaming of LLM responses and tool outputs
- 💾 **Conversation Persistence**: Save and resume conversations
- 🎨 **Multiple Interfaces**: CLI, Web UI, and API endpoints
- 🔧 **Customizable**: Easy to extend with custom tools and providers

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/open-agent-mode.git
cd open-agent-mode

# Install dependencies
pip install -e .

# Or with poetry
poetry install
```

## Quick Start

### CLI Usage

```bash
# Start interactive session
open-agent

# Run with specific model
open-agent --model gpt-4 --provider openai

# Execute single command
open-agent "Create a Python script that fetches weather data"
```

### Web UI

```bash
# Start web server
open-agent serve --port 8000
```

### Python API

```python
from open_agent import Agent, OpenAIProvider

agent = Agent(provider=OpenAIProvider(api_key="your-key"))
response = agent.run("Write a factorial function in Python")
print(response)
```

## Configuration

Create a `.env` file or set environment variables:

```env
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
SEARCH_API_KEY=your-search-api-key
```

## Architecture

```
open-agent-mode/
├── src/
│   ├── open_agent/
│   │   ├── __init__.py
│   │   ├── agent.py          # Core agent logic
│   │   ├── providers/        # LLM providers
│   │   ├── tools/           # Tool implementations
│   │   ├── ui/              # User interfaces
│   │   └── utils/           # Utilities
├── tests/
├── examples/
└── docs/
```

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

## License

MIT License - see [LICENSE](LICENSE) file for details.

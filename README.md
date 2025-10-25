# Open Agent Mode

An open-source implementation of ChatGPT's agent mode with tool use, code execution, and isolated VM execution environment.

## Features

- 🤖 **Multi-LLM Support**: OpenAI, Anthropic Claude, Google Gemini, Groq, Ollama (local), and vLLM
- 🐳 **Isolated VM Environment**: Docker-based containerized execution for safe code running
- 🛠️ **Extensible Tool System**: Built-in tools for file operations, code execution, package management
- 🔄 **Streaming Responses**: Real-time streaming of LLM responses and tool outputs
- 💾 **Conversation Persistence**: Save and resume conversations with VM state snapshots
- 🎨 **Multiple Interfaces**: CLI, Web UI, and Python API
- 🔧 **Customizable**: Easy to extend with custom tools and providers
- 🌐 **Multi-Language**: Execute code in Python, JavaScript, Go, Ruby, Rust, and more

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
response = await agent.run("Write a factorial function in Python")
print(response)
```

### VM-Enabled Agent (Isolated Execution)

```python
import asyncio
from open_agent.agent_vm import VMAgent
from open_agent.providers import OpenAIProvider

async def main():
    provider = OpenAIProvider(api_key="your-key")
    
    # Agent with isolated VM environment
    async with VMAgent(provider=provider) as agent:
        # Agent can safely execute code, install packages, etc.
        response = await agent.run("""
            Create a Flask web application with:
            1. A homepage
            2. A /api/hello endpoint
            3. Proper project structure
            Then show me the files created.
        """)
        print(response)

asyncio.run(main())
```

## Supported AI Providers

### 1. OpenAI (GPT-4, GPT-3.5)

```python
from open_agent import Agent
from open_agent.providers import OpenAIProvider

provider = OpenAIProvider(
    api_key="your-openai-key",  # or set OPENAI_API_KEY
    model="gpt-4-turbo-preview"
)
agent = Agent(provider=provider)
```

**Models**: `gpt-4-turbo-preview`, `gpt-4`, `gpt-3.5-turbo`

### 2. Anthropic Claude

```python
from open_agent.providers import AnthropicProvider

provider = AnthropicProvider(
    api_key="your-anthropic-key",  # or set ANTHROPIC_API_KEY
    model="claude-3-5-sonnet-20241022"
)
agent = Agent(provider=provider)
```

**Models**: `claude-3-5-sonnet-20241022`, `claude-3-opus-20240229`, `claude-3-haiku-20240307`

### 3. Google Gemini

```python
from open_agent.providers import GeminiProvider

provider = GeminiProvider(
    api_key="your-google-key",  # or set GOOGLE_API_KEY
    model="gemini-1.5-pro"
)
agent = Agent(provider=provider)
```

**Models**: `gemini-1.5-pro`, `gemini-1.5-flash`, `gemini-pro`

### 4. Groq (Ultra-fast Inference)

```python
from open_agent.providers import GroqProvider

provider = GroqProvider(
    api_key="your-groq-key",  # or set GROQ_API_KEY
    model="llama-3.1-70b-versatile"
)
agent = Agent(provider=provider)
```

**Models**: `llama-3.1-405b-reasoning`, `llama-3.1-70b-versatile`, `llama-3.1-8b-instant`, `mixtral-8x7b-32768`

### 5. Ollama (Local Models)

```python
from open_agent.providers import OllamaProvider

provider = OllamaProvider(
    model="llama3.1",
    base_url="http://localhost:11434"  # default Ollama endpoint
)
agent = Agent(provider=provider)
```

**Setup**:
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3.1
ollama pull mistral
ollama pull codellama
```

**Models**: Any model from [Ollama Library](https://ollama.com/library)

### 6. vLLM (Coming Soon)

Support for self-hosted vLLM inference servers.

## Environment Variables

Create a `.env` file or set environment variables:

```env
# Choose your provider(s)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AI...
GROQ_API_KEY=gsk_...

# Optional: Ollama endpoint (defaults to localhost:11434)
OLLAMA_BASE_URL=http://localhost:11434
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

#!/usr/bin/env python3
"""Examples using different AI providers."""

import asyncio
import os
from open_agent import Agent, AgentConfig
from open_agent.providers import (
    OpenAIProvider,
    AnthropicProvider,
    GeminiProvider,
    OllamaProvider,
    GroqProvider
)


async def openai_example():
    """Example using OpenAI."""
    print("=== OpenAI Example ===\n")
    
    provider = OpenAIProvider(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4-turbo-preview"
    )
    
    agent = Agent(provider=provider)
    response = await agent.run("Write a haiku about AI")
    print(f"OpenAI: {response}\n")


async def anthropic_example():
    """Example using Anthropic Claude."""
    print("=== Anthropic Claude Example ===\n")
    
    provider = AnthropicProvider(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model="claude-3-5-sonnet-20241022"
    )
    
    agent = Agent(provider=provider)
    response = await agent.run("Write a haiku about AI")
    print(f"Claude: {response}\n")


async def gemini_example():
    """Example using Google Gemini."""
    print("=== Google Gemini Example ===\n")
    
    provider = GeminiProvider(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-1.5-pro"
    )
    
    agent = Agent(provider=provider)
    response = await agent.run("Write a haiku about AI")
    print(f"Gemini: {response}\n")


async def ollama_example():
    """Example using Ollama (local)."""
    print("=== Ollama (Local) Example ===\n")
    
    try:
        provider = OllamaProvider(
            model="llama3.1",
            base_url="http://localhost:11434"
        )
        
        agent = Agent(provider=provider)
        response = await agent.run("Write a haiku about AI")
        print(f"Ollama: {response}\n")
    except Exception as e:
        print(f"Ollama error (is it running?): {e}\n")


async def groq_example():
    """Example using Groq (ultra-fast)."""
    print("=== Groq Example ===\n")
    
    provider = GroqProvider(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-70b-versatile"
    )
    
    agent = Agent(provider=provider)
    response = await agent.run("Write a haiku about AI")
    print(f"Groq: {response}\n")


async def compare_providers():
    """Compare responses from multiple providers."""
    print("=== Comparing All Providers ===\n")
    
    question = "Explain recursion in one sentence"
    
    providers = {}
    
    # OpenAI
    if os.getenv("OPENAI_API_KEY"):
        providers["OpenAI GPT-4"] = OpenAIProvider(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4-turbo-preview"
        )
    
    # Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        providers["Claude 3.5"] = AnthropicProvider(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model="claude-3-5-sonnet-20241022"
        )
    
    # Gemini
    if os.getenv("GOOGLE_API_KEY"):
        providers["Gemini 1.5"] = GeminiProvider(
            api_key=os.getenv("GOOGLE_API_KEY"),
            model="gemini-1.5-pro"
        )
    
    # Groq
    if os.getenv("GROQ_API_KEY"):
        providers["Groq Llama 3.1"] = GroqProvider(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.1-70b-versatile"
        )
    
    # Ollama
    try:
        providers["Ollama (Local)"] = OllamaProvider(
            model="llama3.1",
            base_url="http://localhost:11434"
        )
    except:
        pass
    
    for name, provider in providers.items():
        try:
            agent = Agent(provider=provider)
            response = await agent.run(question)
            print(f"{name}:")
            print(f"  {response}\n")
        except Exception as e:
            print(f"{name}: Error - {e}\n")


async def streaming_example():
    """Example of streaming with different providers."""
    print("=== Streaming Example ===\n")
    
    # Try with available provider
    provider = None
    provider_name = ""
    
    if os.getenv("OPENAI_API_KEY"):
        provider = OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))
        provider_name = "OpenAI"
    elif os.getenv("ANTHROPIC_API_KEY"):
        provider = AnthropicProvider(api_key=os.getenv("ANTHROPIC_API_KEY"))
        provider_name = "Anthropic"
    elif os.getenv("GROQ_API_KEY"):
        provider = GroqProvider(api_key=os.getenv("GROQ_API_KEY"))
        provider_name = "Groq"
    
    if provider:
        config = AgentConfig(stream=True)
        agent = Agent(provider=provider, config=config)
        
        print(f"Streaming from {provider_name}:")
        async for token in agent.stream_response("Count from 1 to 5"):
            print(token, end="", flush=True)
        print("\n")


async def tool_usage_example():
    """Example of tool usage with different providers."""
    print("=== Tool Usage Example ===\n")
    
    # Choose provider based on what's available
    if os.getenv("OPENAI_API_KEY"):
        provider = OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))
        provider_name = "OpenAI"
    elif os.getenv("ANTHROPIC_API_KEY"):
        provider = AnthropicProvider(api_key=os.getenv("ANTHROPIC_API_KEY"))
        provider_name = "Anthropic"
    elif os.getenv("GOOGLE_API_KEY"):
        provider = GeminiProvider(api_key=os.getenv("GOOGLE_API_KEY"))
        provider_name = "Gemini"
    else:
        print("No API key found. Please set an API key.")
        return
    
    agent = Agent(provider=provider)
    
    print(f"Using {provider_name} with tools:")
    response = await agent.run(
        "Create a file called 'test.txt' with the content 'Hello, World!'"
    )
    print(f"Response: {response}\n")


async def main():
    """Run examples."""
    print("=== Open Agent Mode - Multi-Provider Examples ===\n")
    
    # Check for API keys
    available_providers = []
    if os.getenv("OPENAI_API_KEY"):
        available_providers.append("OpenAI")
    if os.getenv("ANTHROPIC_API_KEY"):
        available_providers.append("Anthropic")
    if os.getenv("GOOGLE_API_KEY"):
        available_providers.append("Google Gemini")
    if os.getenv("GROQ_API_KEY"):
        available_providers.append("Groq")
    
    print(f"Available providers: {', '.join(available_providers) if available_providers else 'None'}\n")
    
    if not available_providers:
        print("Please set at least one API key:")
        print("  export OPENAI_API_KEY='your-key'")
        print("  export ANTHROPIC_API_KEY='your-key'")
        print("  export GOOGLE_API_KEY='your-key'")
        print("  export GROQ_API_KEY='your-key'")
        return
    
    print("Select example:")
    print("1. OpenAI")
    print("2. Anthropic Claude")
    print("3. Google Gemini")
    print("4. Ollama (Local)")
    print("5. Groq")
    print("6. Compare all providers")
    print("7. Streaming example")
    print("8. Tool usage example")
    print("9. Run all examples")
    
    choice = input("\nEnter choice (1-9): ")
    
    examples = {
        "1": openai_example,
        "2": anthropic_example,
        "3": gemini_example,
        "4": ollama_example,
        "5": groq_example,
        "6": compare_providers,
        "7": streaming_example,
        "8": tool_usage_example
    }
    
    if choice == "9":
        for example in examples.values():
            try:
                await example()
                print("\n" + "="*50 + "\n")
            except Exception as e:
                print(f"Error: {e}\n")
    elif choice in examples:
        await examples[choice]()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    asyncio.run(main())
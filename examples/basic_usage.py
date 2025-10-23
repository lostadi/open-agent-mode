#!/usr/bin/env python3
"""Basic usage examples for Open Agent Mode."""

import asyncio
import os
from pathlib import Path
from open_agent import Agent, AgentConfig, OpenAIProvider
from open_agent.tools import tool


# Example 1: Simple usage with OpenAI
async def simple_example():
    """Simple example using OpenAI provider."""
    print("=== Simple Example ===\n")
    
    # Create provider
    provider = OpenAIProvider(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4-turbo-preview"
    )
    
    # Create agent
    agent = Agent(provider=provider)
    
    # Run a simple query
    response = await agent.run("What is the capital of France?")
    print(f"Response: {response}\n")


# Example 2: Using tools
async def tools_example():
    """Example using built-in tools."""
    print("=== Tools Example ===\n")
    
    provider = OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))
    agent = Agent(provider=provider)
    
    # Ask agent to create a file
    response = await agent.run(
        "Create a Python file called 'hello.py' that prints 'Hello, World!'"
    )
    print(f"Response: {response}\n")
    
    # Ask agent to read and execute the file
    response = await agent.run("Now read the hello.py file and execute it")
    print(f"Response: {response}\n")


# Example 3: Custom tools
@tool(name="get_weather", description="Get current weather for a city")
def get_weather(city: str, unit: str = "celsius") -> str:
    """Mock weather function."""
    # This would normally call a weather API
    return f"The weather in {city} is sunny and 22°{unit[0].upper()}"


@tool(name="calculate", description="Perform mathematical calculations")
def calculate(expression: str) -> float:
    """Safe math evaluation."""
    # In production, use a proper math parser
    allowed_names = {
        k: v for k, v in __builtins__.items()
        if k in ["abs", "round", "min", "max", "sum"]
    }
    return eval(expression, {"__builtins__": {}}, allowed_names)


async def custom_tools_example():
    """Example with custom tools."""
    print("=== Custom Tools Example ===\n")
    
    provider = OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))
    
    config = AgentConfig(
        tools_enabled=True,
        temperature=0.7
    )
    
    agent = Agent(provider=provider, config=config)
    
    # Tools are auto-registered via decorator
    response = await agent.run("What's the weather in Paris?")
    print(f"Weather response: {response}\n")
    
    response = await agent.run("Calculate 15 * 37 + 129")
    print(f"Calculation response: {response}\n")


# Example 4: Conversation management
async def conversation_example():
    """Example of conversation management."""
    print("=== Conversation Example ===\n")
    
    provider = OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))
    
    config = AgentConfig(
        auto_save=True,
        save_path=Path("conversation.json")
    )
    
    agent = Agent(provider=provider, config=config)
    
    # Have a multi-turn conversation
    await agent.run("My name is Alice. Remember that.")
    response = await agent.run("What's my name?")
    print(f"Agent remembers: {response}\n")
    
    # Clear and verify
    agent.clear_conversation()
    response = await agent.run("What's my name?")
    print(f"After clear: {response}\n")
    
    # Load previous conversation
    agent.load_conversation(Path("conversation.json"))
    response = await agent.run("What's my name again?")
    print(f"After loading: {response}\n")


# Example 5: Streaming responses
async def streaming_example():
    """Example of streaming responses."""
    print("=== Streaming Example ===\n")
    
    provider = OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))
    
    config = AgentConfig(stream=True)
    agent = Agent(provider=provider, config=config)
    
    print("Streaming response: ", end="", flush=True)
    async for token in agent.stream_response(
        "Write a haiku about programming"
    ):
        print(token, end="", flush=True)
    print("\n")


# Example 6: System prompts
async def system_prompt_example():
    """Example using system prompts."""
    print("=== System Prompt Example ===\n")
    
    provider = OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))
    agent = Agent(provider=provider)
    
    system_prompt = """You are a helpful Python programming assistant. 
    Always provide code examples when explaining concepts.
    Use type hints in your Python code."""
    
    response = await agent.run(
        "Explain list comprehensions",
        system_prompt=system_prompt
    )
    print(f"Response with system prompt:\n{response}\n")


async def main():
    """Run all examples."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    try:
        await simple_example()
        await tools_example()
        await custom_tools_example()
        await conversation_example()
        await streaming_example()
        await system_prompt_example()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())

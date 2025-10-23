#!/usr/bin/env python3
"""Example demonstrating the VM-enabled agent capabilities."""

import asyncio
import os
from open_agent import OpenAIProvider
from open_agent.agent_vm import VMAgent, VMAgentSession, VMConfig
from open_agent import AgentConfig


async def basic_vm_example():
    """Basic VM agent example."""
    print("=== Basic VM Agent Example ===\n")
    
    # Configure provider
    provider = OpenAIProvider(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4-turbo-preview"
    )
    
    # Configure VM
    vm_config = VMConfig(
        memory_limit="2g",
        cpu_limit=2.0,
        persist_data=True
    )
    
    # Create VM agent
    async with VMAgent(provider=provider, vm_config=vm_config) as agent:
        # The VM is automatically initialized
        
        # Get VM status
        status = await agent.get_vm_status()
        print(f"VM Status: {status['status']}")
        print(f"Session ID: {status['session_id']}\n")
        
        # Ask agent to create and run a Python script
        response = await agent.run("""
        Create a Python script that:
        1. Generates 100 random numbers
        2. Calculates their mean and standard deviation
        3. Creates a histogram plot
        4. Saves the plot as 'histogram.png'
        Then run the script and show me the results.
        """)
        print(f"Agent: {response}\n")
        
        # Ask agent to analyze the results
        response = await agent.run("What files were created? Show me their contents.")
        print(f"Agent: {response}\n")


async def web_development_example():
    """Web development in VM example."""
    print("=== Web Development Example ===\n")
    
    provider = OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))
    
    async with VMAgentSession(provider) as agent:
        # Create a web project
        response = await agent.run("""
        Create a simple Flask web application that:
        1. Has a homepage with a form
        2. Accepts a user's name
        3. Shows a personalized greeting
        4. Has some basic CSS styling
        
        Set it up in a project called 'greeting-app'
        """)
        print(f"Project created: {response}\n")
        
        # Test the application
        response = await agent.run("""
        Now:
        1. Install Flask if needed
        2. Run the Flask application
        3. Show me the file structure
        4. Explain how to access it
        """)
        print(f"Application running: {response}\n")


async def data_science_example():
    """Data science workflow example."""
    print("=== Data Science Example ===\n")
    
    provider = OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))
    vm_config = VMConfig(memory_limit="4g")
    
    agent = VMAgent(provider=provider, vm_config=vm_config)
    await agent.initialize_vm()
    
    try:
        # Create a data science project
        response = await agent.run("""
        Create a data science project that:
        1. Generates a synthetic dataset with 1000 samples
        2. Has features: age, income, education_years, job_satisfaction
        3. Performs exploratory data analysis
        4. Trains a simple prediction model
        5. Evaluates the model performance
        
        Save all outputs and visualizations.
        """)
        print(f"Data science project: {response}\n")
        
        # Create checkpoint
        checkpoint_id = await agent.create_checkpoint("after_analysis")
        print(f"Checkpoint created: {checkpoint_id}\n")
        
        # Modify the project
        response = await agent.run("""
        Now add feature engineering:
        1. Create interaction features
        2. Normalize the data
        3. Re-train the model
        4. Compare results
        """)
        print(f"Enhanced model: {response}\n")
        
        # Optionally restore to checkpoint
        # await agent.restore_checkpoint(checkpoint_id)
        
    finally:
        await agent.cleanup()


async def multi_language_example():
    """Multi-language programming example."""
    print("=== Multi-Language Example ===\n")
    
    provider = OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))
    
    async with VMAgent(provider=provider) as agent:
        # Create programs in different languages
        response = await agent.run("""
        Create a simple 'FizzBuzz' implementation in:
        1. Python (fizzbuzz.py)
        2. JavaScript (fizzbuzz.js)
        3. Go (fizzbuzz.go)
        4. Ruby (fizzbuzz.rb)
        
        Then run each one and compare their outputs.
        """)
        print(f"Multi-language FizzBuzz: {response}\n")
        
        # Benchmark them
        response = await agent.run("""
        Now create a benchmark script that:
        1. Times each implementation
        2. Compares their performance
        3. Creates a summary table
        """)
        print(f"Benchmark results: {response}\n")


async def package_management_example():
    """Package installation and management example."""
    print("=== Package Management Example ===\n")
    
    provider = OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))
    
    async with VMAgent(provider=provider) as agent:
        # Install packages
        packages = ["fastapi", "uvicorn", "sqlalchemy", "alembic"]
        results = await agent.install_in_vm(packages, manager="pip")
        print(f"Installed packages: {[r['package'] for r in results if r['success']]}\n")
        
        # Create API project using installed packages
        response = await agent.run("""
        Using the installed packages, create a REST API that:
        1. Has a SQLite database
        2. Manages a simple TODO list
        3. Has CRUD endpoints
        4. Includes database migrations
        
        Set it up properly with all necessary files.
        """)
        print(f"API created: {response}\n")


async def interactive_session():
    """Interactive VM session with the agent."""
    print("=== Interactive VM Session ===\n")
    print("Type 'exit' to quit, 'status' for VM info, or your requests:\n")
    
    provider = OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))
    
    async with VMAgent(provider=provider) as agent:
        while True:
            try:
                user_input = input("\nYou: ")
                
                if user_input.lower() == 'exit':
                    break
                elif user_input.lower() == 'status':
                    status = await agent.get_vm_status()
                    print(f"VM Status: {status}")
                elif user_input.lower() == 'checkpoint':
                    checkpoint_id = await agent.create_checkpoint()
                    print(f"Checkpoint created: {checkpoint_id}")
                else:
                    response = await agent.run(user_input)
                    print(f"\nAgent: {response}")
                    
            except KeyboardInterrupt:
                print("\nUse 'exit' to quit properly")
            except Exception as e:
                print(f"Error: {e}")


async def main():
    """Run examples."""
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    # Check Docker availability
    try:
        import docker
        client = docker.from_env()
        client.ping()
    except Exception as e:
        print(f"Docker not available: {e}")
        print("Please ensure Docker is installed and running")
        return
    
    print("Select an example:")
    print("1. Basic VM operations")
    print("2. Web development")
    print("3. Data science workflow")
    print("4. Multi-language programming")
    print("5. Package management")
    print("6. Interactive session")
    print("7. Run all examples")
    
    choice = input("\nEnter choice (1-7): ")
    
    examples = {
        "1": basic_vm_example,
        "2": web_development_example,
        "3": data_science_example,
        "4": multi_language_example,
        "5": package_management_example,
        "6": interactive_session
    }
    
    if choice == "7":
        for example in list(examples.values())[:-1]:  # Skip interactive
            await example()
            print("\n" + "="*50 + "\n")
    elif choice in examples:
        await examples[choice]()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    asyncio.run(main())

"""Command-line interface for Open Agent Mode."""

import asyncio
import click
import sys
from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.panel import Panel
from rich.syntax import Syntax
import logging
from typing import Optional

from .agent import Agent, AgentConfig
from .providers.openai_provider import OpenAIProvider


console = Console()
logging.basicConfig(level=logging.INFO)


@click.group(invoke_without_command=True)
@click.pass_context
@click.option('--provider', default='openai', help='LLM provider to use')
@click.option('--model', default=None, help='Model to use')
@click.option('--api-key', envvar='OPENAI_API_KEY', help='API key for provider')
@click.option('--temperature', default=0.7, help='Sampling temperature')
@click.option('--max-tokens', default=None, type=int, help='Maximum tokens to generate')
@click.option('--no-tools', is_flag=True, help='Disable tool use')
@click.option('--save-path', type=Path, help='Path to save conversations')
@click.option('--load-path', type=Path, help='Path to load conversation from')
@click.option('--system-prompt', help='System prompt to use')
@click.option('--verbose', is_flag=True, help='Enable verbose logging')
def cli(ctx, provider, model, api_key, temperature, max_tokens, 
        no_tools, save_path, load_path, system_prompt, verbose):
    """Open Agent Mode - ChatGPT-like agent with tool use.
    
    Run without arguments for interactive mode, or use subcommands.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # If no subcommand, run interactive mode
    if ctx.invoked_subcommand is None:
        asyncio.run(interactive_mode(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            tools_enabled=not no_tools,
            save_path=save_path,
            load_path=load_path,
            system_prompt=system_prompt
        ))


@cli.command()
@click.argument('message')
@click.option('--provider', default='openai', help='LLM provider to use')
@click.option('--model', default=None, help='Model to use')
@click.option('--api-key', envvar='OPENAI_API_KEY', help='API key for provider')
@click.option('--temperature', default=0.7, help='Sampling temperature')
@click.option('--max-tokens', default=None, type=int, help='Maximum tokens')
@click.option('--no-tools', is_flag=True, help='Disable tool use')
@click.option('--system-prompt', help='System prompt to use')
def run(message, provider, model, api_key, temperature, 
        max_tokens, no_tools, system_prompt):
    """Run a single message and exit."""
    asyncio.run(single_message(
        message=message,
        provider=provider,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        tools_enabled=not no_tools,
        system_prompt=system_prompt
    ))


@cli.command()
@click.option('--host', default='127.0.0.1', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
def serve(host, port, reload):
    """Start the web UI server."""
    console.print(f"[green]Starting web server on http://{host}:{port}[/green]")
    
    try:
        import uvicorn
        uvicorn.run(
            "open_agent.web:app",
            host=host,
            port=port,
            reload=reload
        )
    except ImportError:
        console.print("[red]Web server dependencies not installed. Install with: pip install open-agent-mode[web][/red]")
        sys.exit(1)


async def interactive_mode(provider: str, model: Optional[str], api_key: str,
                          temperature: float, max_tokens: Optional[int],
                          tools_enabled: bool, save_path: Optional[Path],
                          load_path: Optional[Path], system_prompt: Optional[str]):
    """Run interactive chat mode."""
    console.print(Panel.fit(
        "[bold cyan]Open Agent Mode[/bold cyan]\n"
        "Type your message and press Enter. Use /help for commands.",
        border_style="cyan"
    ))
    
    # Create provider
    if provider == "openai":
        if not api_key:
            console.print("[red]OpenAI API key required. Set OPENAI_API_KEY or use --api-key[/red]")
            return
        
        llm_provider = OpenAIProvider(
            api_key=api_key,
            model=model or "gpt-4-turbo-preview"
        )
    else:
        console.print(f"[red]Unknown provider: {provider}[/red]")
        return
    
    # Create agent
    config = AgentConfig(
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        tools_enabled=tools_enabled,
        auto_save=save_path is not None,
        save_path=save_path,
        stream=True
    )
    
    agent = Agent(provider=llm_provider, config=config)
    
    # Load conversation if specified
    if load_path and load_path.exists():
        agent.load_conversation(load_path)
        console.print(f"[green]Loaded conversation from {load_path}[/green]")
    
    # Main loop
    while True:
        try:
            # Get user input
            user_input = Prompt.ask("\n[bold blue]You[/bold blue]")
            
            # Handle commands
            if user_input.startswith("/"):
                if handle_command(user_input, agent):
                    continue
                else:
                    break
            
            # Process message
            console.print("\n[bold green]Assistant:[/bold green]")
            
            # Stream response
            response = ""
            async for token in agent.stream_response(user_input, system_prompt):
                response += token
                console.print(token, end="")
            
            console.print()  # New line after response
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Use /exit to quit[/yellow]")
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            if logging.getLogger().level == logging.DEBUG:
                import traceback
                traceback.print_exc()


async def single_message(message: str, provider: str, model: Optional[str],
                        api_key: str, temperature: float, max_tokens: Optional[int],
                        tools_enabled: bool, system_prompt: Optional[str]):
    """Process a single message and exit."""
    # Create provider
    if provider == "openai":
        if not api_key:
            console.print("[red]OpenAI API key required. Set OPENAI_API_KEY or use --api-key[/red]")
            return
        
        llm_provider = OpenAIProvider(
            api_key=api_key,
            model=model or "gpt-4-turbo-preview"
        )
    else:
        console.print(f"[red]Unknown provider: {provider}[/red]")
        return
    
    # Create agent
    config = AgentConfig(
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        tools_enabled=tools_enabled,
        stream=True
    )
    
    agent = Agent(provider=llm_provider, config=config)
    
    # Process message
    response = ""
    async for token in agent.stream_response(message, system_prompt):
        response += token
        console.print(token, end="")
    
    console.print()  # New line at end


def handle_command(command: str, agent: Agent) -> bool:
    """Handle CLI commands.
    
    Returns:
        True to continue, False to exit
    """
    cmd = command.lower().strip()
    
    if cmd in ["/exit", "/quit", "/q"]:
        console.print("[yellow]Goodbye![/yellow]")
        return False
    
    elif cmd == "/help":
        help_text = """
[bold]Available Commands:[/bold]
  /help     - Show this help message
  /clear    - Clear conversation history
  /save     - Save conversation
  /load     - Load conversation
  /tools    - List available tools
  /exit     - Exit the program
        """
        console.print(Panel(help_text, title="Help", border_style="blue"))
    
    elif cmd == "/clear":
        agent.clear_conversation()
        console.print("[green]Conversation cleared[/green]")
    
    elif cmd.startswith("/save"):
        parts = cmd.split(maxsplit=1)
        if len(parts) > 1:
            path = Path(parts[1])
            agent.save_conversation(path)
            console.print(f"[green]Conversation saved to {path}[/green]")
        else:
            console.print("[red]Usage: /save <path>[/red]")
    
    elif cmd.startswith("/load"):
        parts = cmd.split(maxsplit=1)
        if len(parts) > 1:
            path = Path(parts[1])
            if path.exists():
                agent.load_conversation(path)
                console.print(f"[green]Conversation loaded from {path}[/green]")
            else:
                console.print(f"[red]File not found: {path}[/red]")
        else:
            console.print("[red]Usage: /load <path>[/red]")
    
    elif cmd == "/tools":
        tools = agent.tool_registry.list()
        if tools:
            console.print("[bold]Available Tools:[/bold]")
            for tool in tools:
                console.print(f"  • {tool}")
        else:
            console.print("[yellow]No tools available[/yellow]")
    
    else:
        console.print(f"[red]Unknown command: {command}[/red]")
        console.print("Type /help for available commands")
    
    return True


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()

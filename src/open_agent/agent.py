"""Main agent implementation with conversation management and tool execution."""

import json
import logging
from typing import List, Dict, Any, Optional, AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from .tools.base import ToolRegistry, ToolResult
from .tools.file_tools import (
    ReadFileTool, WriteFileTool, AppendFileTool,
    ListDirectoryTool, DeleteFileTool, MoveFileTool, CopyFileTool
)
from .tools.code_tools import (
    ExecuteCodeTool, RunCommandTool, FormatCodeTool, AnalyzeCodeTool
)


logger = logging.getLogger(__name__)


class Message(BaseModel):
    """A single message in the conversation."""
    role: str  # "user", "assistant", "system", "tool"
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Conversation(BaseModel):
    """Manages conversation history and context."""
    messages: List[Message] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def add_message(self, message: Message):
        """Add a message to the conversation."""
        self.messages.append(message)
        self.updated_at = datetime.now()
    
    def get_messages(self, limit: Optional[int] = None) -> List[Message]:
        """Get messages with optional limit."""
        if limit:
            return self.messages[-limit:]
        return self.messages
    
    def clear(self):
        """Clear all messages."""
        self.messages.clear()
        self.updated_at = datetime.now()
    
    def to_openai_format(self) -> List[Dict[str, Any]]:
        """Convert to OpenAI chat format."""
        formatted = []
        for msg in self.messages:
            if msg.role == "tool":
                formatted.append({
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id
                })
            elif msg.tool_calls:
                formatted.append({
                    "role": msg.role,
                    "content": msg.content,
                    "tool_calls": msg.tool_calls
                })
            else:
                formatted.append({
                    "role": msg.role,
                    "content": msg.content
                })
        return formatted
    
    def save(self, path: Path):
        """Save conversation to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.model_dump(), f, indent=2, default=str)
    
    @classmethod
    def load(cls, path: Path) -> "Conversation":
        """Load conversation from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Convert timestamp strings back to datetime
        for msg in data.get('messages', []):
            if 'timestamp' in msg:
                msg['timestamp'] = datetime.fromisoformat(msg['timestamp'])
        
        if 'created_at' in data:
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data:
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        
        return cls(**data)


@dataclass
class AgentConfig:
    """Configuration for the agent."""
    provider: Optional[str] = "openai"
    model: Optional[str] = "gpt-4"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    tools_enabled: bool = True
    auto_save: bool = True
    save_path: Optional[Path] = None
    max_retries: int = 3
    timeout: int = 30
    stream: bool = True


class Agent:
    """Main agent class for handling conversations and tool execution."""
    
    def __init__(self, 
                 provider=None,
                 config: Optional[AgentConfig] = None):
        """Initialize the agent.
        
        Args:
            provider: LLM provider instance
            config: Agent configuration
        """
        self.provider = provider
        self.config = config or AgentConfig()
        self.conversation = Conversation()
        self.tool_registry = ToolRegistry()
        
        # Register default tools
        if self.config.tools_enabled:
            self._register_default_tools()
        
        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _register_default_tools(self):
        """Register default built-in tools."""
        # File tools
        self.tool_registry.register(ReadFileTool())
        self.tool_registry.register(WriteFileTool())
        self.tool_registry.register(AppendFileTool())
        self.tool_registry.register(ListDirectoryTool())
        self.tool_registry.register(DeleteFileTool())
        self.tool_registry.register(MoveFileTool())
        self.tool_registry.register(CopyFileTool())
        
        # Code tools
        self.tool_registry.register(ExecuteCodeTool())
        self.tool_registry.register(RunCommandTool())
        self.tool_registry.register(FormatCodeTool())
        self.tool_registry.register(AnalyzeCodeTool())
    
    def register_tool(self, tool):
        """Register a custom tool."""
        self.tool_registry.register(tool)
    
    async def process_message(self, message: str, 
                             system_prompt: Optional[str] = None) -> str:
        """Process a user message and return response.
        
        Args:
            message: User message
            system_prompt: Optional system prompt to prepend
        
        Returns:
            Assistant's response
        """
        # Add system prompt if provided and not already present
        if system_prompt and (not self.conversation.messages or 
                             self.conversation.messages[0].role != "system"):
            self.conversation.add_message(Message(
                role="system",
                content=system_prompt
            ))
        
        # Add user message
        self.conversation.add_message(Message(
            role="user",
            content=message
        ))
        
        # Get response from provider
        response = await self._get_llm_response()
        
        # Auto-save if enabled
        if self.config.auto_save and self.config.save_path:
            self.conversation.save(self.config.save_path)
        
        return response
    
    async def _get_llm_response(self) -> str:
        """Get response from LLM provider."""
        if not self.provider:
            raise ValueError("No LLM provider configured")
        
        messages = self.conversation.to_openai_format()
        tools = self.tool_registry.get_openai_tools() if self.config.tools_enabled else None
        
        # Get completion from provider
        response = await self.provider.get_completion(
            messages=messages,
            tools=tools,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=self.config.stream
        )
        
        # Handle streaming response
        if self.config.stream:
            full_response = ""
            tool_calls = []
            
            async for chunk in response:
                if chunk.get("content"):
                    full_response += chunk["content"]
                if chunk.get("tool_calls"):
                    tool_calls.extend(chunk["tool_calls"])
            
            response = {
                "content": full_response,
                "tool_calls": tool_calls if tool_calls else None
            }
        
        # Process tool calls if present
        if response.get("tool_calls"):
            tool_results = await self._execute_tools(response["tool_calls"])
            
            # Add assistant message with tool calls
            self.conversation.add_message(Message(
                role="assistant",
                content=response.get("content", ""),
                tool_calls=response["tool_calls"]
            ))
            
            # Add tool results
            for tool_call, result in zip(response["tool_calls"], tool_results):
                self.conversation.add_message(Message(
                    role="tool",
                    content=json.dumps(result.model_dump()),
                    tool_call_id=tool_call.get("id")
                ))
            
            # Get final response after tool execution
            return await self._get_llm_response()
        else:
            # Add assistant response
            self.conversation.add_message(Message(
                role="assistant",
                content=response.get("content", "")
            ))
            
            return response.get("content", "")
    
    async def _execute_tools(self, tool_calls: List[Dict]) -> List[ToolResult]:
        """Execute tool calls and return results."""
        results = []
        
        for call in tool_calls:
            function = call.get("function", {})
            name = function.get("name")
            
            try:
                arguments = json.loads(function.get("arguments", "{}"))
            except json.JSONDecodeError:
                arguments = {}
            
            self.logger.info(f"Executing tool: {name} with args: {arguments}")
            
            result = await self.tool_registry.execute(name, **arguments)
            results.append(result)
            
            self.logger.info(f"Tool result: {result.success}")
        
        return results
    
    async def stream_response(self, message: str, 
                            system_prompt: Optional[str] = None) -> AsyncIterator[str]:
        """Stream response tokens as they arrive.
        
        Args:
            message: User message
            system_prompt: Optional system prompt
        
        Yields:
            Response tokens
        """
        # Add system prompt if provided
        if system_prompt and (not self.conversation.messages or 
                             self.conversation.messages[0].role != "system"):
            self.conversation.add_message(Message(
                role="system",
                content=system_prompt
            ))
        
        # Add user message
        self.conversation.add_message(Message(
            role="user",
            content=message
        ))
        
        # Stream from provider
        messages = self.conversation.to_openai_format()
        tools = self.tool_registry.get_openai_tools() if self.config.tools_enabled else None
        
        stream = await self.provider.stream_completion(
            messages=messages,
            tools=tools,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        full_response = ""
        tool_calls = []
        
        async for chunk in stream:
            if chunk.get("content"):
                full_response += chunk["content"]
                yield chunk["content"]
            
            if chunk.get("tool_calls"):
                tool_calls.extend(chunk["tool_calls"])
        
        # Handle tool calls if present
        if tool_calls:
            # Execute tools
            tool_results = await self._execute_tools(tool_calls)
            
            # Add messages
            self.conversation.add_message(Message(
                role="assistant",
                content=full_response,
                tool_calls=tool_calls
            ))
            
            for tool_call, result in zip(tool_calls, tool_results):
                self.conversation.add_message(Message(
                    role="tool",
                    content=json.dumps(result.model_dump()),
                    tool_call_id=tool_call.get("id")
                ))
            
            # Get follow-up response
            async for token in self.stream_response("", None):
                yield token
        else:
            # Add assistant response
            self.conversation.add_message(Message(
                role="assistant",
                content=full_response
            ))
        
        # Auto-save if enabled
        if self.config.auto_save and self.config.save_path:
            self.conversation.save(self.config.save_path)
    
    def clear_conversation(self):
        """Clear the conversation history."""
        self.conversation.clear()
    
    def save_conversation(self, path: Path):
        """Save conversation to file."""
        self.conversation.save(path)
    
    def load_conversation(self, path: Path):
        """Load conversation from file."""
        self.conversation = Conversation.load(path)
    
    async def run(self, message: str, system_prompt: Optional[str] = None) -> str:
        """Run a single message and return response.
        
        This is the main entry point for simple usage.
        """
        return await self.process_message(message, system_prompt)

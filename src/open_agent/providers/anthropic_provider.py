"""Anthropic Claude provider implementation."""

import os
import json
from typing import Dict, List, Any, Optional, AsyncIterator
from anthropic import AsyncAnthropic
import logging

from .base import LLMProvider


logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider."""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "claude-3-5-sonnet-20241022",
                 base_url: Optional[str] = None):
        """Initialize Anthropic provider.
        
        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Model to use
            base_url: Custom base URL for API
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.base_url = base_url
        
        if not self.api_key:
            raise ValueError("Anthropic API key is required")
        
        self.client = AsyncAnthropic(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    async def get_completion(self,
                            messages: List[Dict[str, Any]],
                            tools: Optional[List[Dict]] = None,
                            temperature: float = 0.7,
                            max_tokens: Optional[int] = None,
                            stream: bool = False,
                            **kwargs) -> Any:
        """Get completion from Anthropic.
        
        Args:
            messages: Conversation messages
            tools: Available tools/functions
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional parameters
        
        Returns:
            Completion response or stream
        """
        try:
            # Convert OpenAI format to Anthropic format
            system_message = ""
            anthropic_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                elif msg["role"] == "tool":
                    # Convert tool result
                    anthropic_messages.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": msg.get("tool_call_id"),
                            "content": msg["content"]
                        }]
                    })
                else:
                    anthropic_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            params = {
                "model": self.model,
                "messages": anthropic_messages,
                "temperature": temperature,
                "max_tokens": max_tokens or 4096,
            }
            
            if system_message:
                params["system"] = system_message
            
            if tools:
                # Convert to Anthropic tool format
                params["tools"] = self._convert_tools(tools)
            
            params.update(kwargs)
            
            if stream:
                return self._stream_completion(params)
            else:
                response = await self.client.messages.create(**params)
                return self._format_response(response)
        
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
    
    def _convert_tools(self, tools: List[Dict]) -> List[Dict]:
        """Convert OpenAI tool format to Anthropic format."""
        anthropic_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func["description"],
                    "input_schema": func["parameters"]
                })
        return anthropic_tools
    
    async def _stream_completion(self, params: Dict) -> AsyncIterator[Dict]:
        """Stream completion from Anthropic."""
        params["stream"] = True
        
        try:
            async with self.client.messages.stream(**params) as stream:
                async for event in stream:
                    result = {}
                    
                    if hasattr(event, 'delta'):
                        if hasattr(event.delta, 'text'):
                            result["content"] = event.delta.text
                    
                    if hasattr(event, 'content_block'):
                        if event.content_block.type == "tool_use":
                            result["tool_calls"] = [{
                                "id": event.content_block.id,
                                "type": "function",
                                "function": {
                                    "name": event.content_block.name,
                                    "arguments": json.dumps(event.content_block.input)
                                }
                            }]
                    
                    if result:
                        yield result
        
        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            raise
    
    def _format_response(self, response) -> Dict[str, Any]:
        """Format Anthropic response to standard format."""
        result = {
            "content": "",
            "role": "assistant"
        }
        
        for block in response.content:
            if block.type == "text":
                result["content"] += block.text
            elif block.type == "tool_use":
                if "tool_calls" not in result:
                    result["tool_calls"] = []
                result["tool_calls"].append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input)
                    }
                })
        
        return result
    
    async def stream_completion(self,
                              messages: List[Dict[str, Any]],
                              tools: Optional[List[Dict]] = None,
                              temperature: float = 0.7,
                              max_tokens: Optional[int] = None,
                              **kwargs) -> AsyncIterator[Dict]:
        """Stream completion tokens."""
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,
            "stream": True
        }
        
        if tools:
            params["tools"] = self._convert_tools(tools)
        
        params.update(kwargs)
        
        async for chunk in self._stream_completion(params):
            yield chunk
    
    def get_max_tokens(self) -> int:
        """Get maximum tokens for model."""
        model_limits = {
            "claude-3-opus": 200000,
            "claude-3-sonnet": 200000,
            "claude-3-haiku": 200000,
            "claude-3-5-sonnet": 200000,
            "claude-2": 100000,
        }
        
        for model_name, limit in model_limits.items():
            if model_name in self.model:
                return limit
        
        return 100000  # Default
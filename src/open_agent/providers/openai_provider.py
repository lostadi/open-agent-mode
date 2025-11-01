"""OpenAI provider implementation."""

import os
from typing import Dict, List, Any, Optional, AsyncIterator
from openai import AsyncOpenAI
import logging

from .base import LLMProvider


logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "gpt-4-turbo-preview",
                 organization: Optional[str] = None,
                 base_url: Optional[str] = None):
        """Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use
            organization: OpenAI organization ID
            base_url: Custom base URL for API
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.organization = organization
        self.base_url = base_url
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            organization=self.organization,
            base_url=self.base_url
        )
    
    async def get_completion(self,
                            messages: List[Dict[str, Any]],
                            tools: Optional[List[Dict]] = None,
                            temperature: float = 0.7,
                            max_tokens: Optional[int] = None,
                            stream: bool = False,
                            **kwargs) -> Any:
        """Get completion from OpenAI.
        
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
            params = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            }
            
            if max_tokens:
                params["max_tokens"] = max_tokens
            
            if tools:
                params["tools"] = tools
                params["tool_choice"] = "auto"
            
            # Add any additional parameters
            params.update(kwargs)
            
            if stream:
                return self._stream_completion(params)
            else:
                response = await self.client.chat.completions.create(**params)
                return self._format_response(response)
        
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    async def _stream_completion(self, params: Dict) -> AsyncIterator[Dict]:
        """Stream completion from OpenAI."""
        params["stream"] = True
        
        try:
            stream = await self.client.chat.completions.create(**params)
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta
                    
                    result = {}
                    
                    if delta.content:
                        result["content"] = delta.content
                    
                    if delta.tool_calls:
                        result["tool_calls"] = []
                        for tool_call in delta.tool_calls:
                            tc = {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.function.name if tool_call.function else None,
                                    "arguments": tool_call.function.arguments if tool_call.function else None
                                }
                            }
                            result["tool_calls"].append(tc)
                    
                    if result:
                        yield result
        
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise
    
    def _format_response(self, response) -> Dict[str, Any]:
        """Format OpenAI response to standard format."""
        message = response.choices[0].message
        
        result = {
            "content": message.content,
            "role": message.role
        }
        
        if message.tool_calls:
            result["tool_calls"] = []
            for tool_call in message.tool_calls:
                tc = {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                }
                result["tool_calls"].append(tc)
        
        return result
    
    async def stream_completion(self,
                              messages: List[Dict[str, Any]],
                              tools: Optional[List[Dict]] = None,
                              temperature: float = 0.7,
                              max_tokens: Optional[int] = None,
                              **kwargs) -> AsyncIterator[Dict]:
        """Stream completion tokens.
        
        Args:
            messages: Conversation messages
            tools: Available tools/functions
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
        
        Yields:
            Response chunks
        """
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True
        }
        
        if max_tokens:
            params["max_tokens"] = max_tokens
        
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"
        
        params.update(kwargs)
        
        async for chunk in self._stream_completion(params):
            yield chunk
    
    def get_token_count(self, text: str) -> int:
        """Get token count for text.
        
        Args:
            text: Text to count tokens for
        
        Returns:
            Token count
        """
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(self.model)
            return len(encoding.encode(text))
        except Exception:
            # Rough estimate if tiktoken fails
            return len(text) // 4
    
    def get_max_tokens(self) -> int:
        """Get maximum tokens for model.
        
        Returns:
            Maximum token limit
        """
        model_limits = {
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-turbo-preview": 128000,
            "gpt-4-1106-preview": 128000,
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384
        }
        
        for model_name, limit in model_limits.items():
            if model_name in self.model:
                return limit
        
        return 4096  # Default

"""Ollama provider for running local LLMs."""

import os
import json
from typing import Dict, List, Any, Optional, AsyncIterator
import aiohttp
import logging

from .base import LLMProvider


logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider."""
    
    def __init__(self, 
                 model: str = "llama3.1",
                 base_url: str = "http://localhost:11434"):
        """Initialize Ollama provider.
        
        Args:
            model: Model to use (e.g., llama3.1, mistral, codellama)
            base_url: Ollama API base URL
        """
        self.model = model
        self.base_url = base_url.rstrip('/')
    
    async def get_completion(self,
                            messages: List[Dict[str, Any]],
                            tools: Optional[List[Dict]] = None,
                            temperature: float = 0.7,
                            max_tokens: Optional[int] = None,
                            stream: bool = False,
                            **kwargs) -> Any:
        """Get completion from Ollama.
        
        Args:
            messages: Conversation messages
            tools: Available tools/functions (limited support)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional parameters
        
        Returns:
            Completion response or stream
        """
        try:
            # Convert messages to Ollama format
            prompt = self._messages_to_prompt(messages)
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "temperature": temperature,
                }
            }
            
            if max_tokens:
                payload["options"]["num_predict"] = max_tokens
            
            payload["options"].update(kwargs)
            
            if stream:
                return self._stream_completion(payload)
            else:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.base_url}/api/generate",
                        json=payload
                    ) as response:
                        result = await response.json()
                        return self._format_response(result)
        
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise
    
    def _messages_to_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """Convert messages to a single prompt string."""
        prompt_parts = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            elif role == "tool":
                prompt_parts.append(f"Tool Result: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)
    
    async def _stream_completion(self, payload: Dict) -> AsyncIterator[Dict]:
        """Stream completion from Ollama."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                ) as response:
                    async for line in response.content:
                        if line:
                            try:
                                data = json.loads(line)
                                if "response" in data:
                                    yield {
                                        "content": data["response"],
                                        "done": data.get("done", False)
                                    }
                            except json.JSONDecodeError:
                                continue
        
        except Exception as e:
            logger.error(f"Ollama streaming error: {e}")
            raise
    
    def _format_response(self, response: Dict) -> Dict[str, Any]:
        """Format Ollama response to standard format."""
        return {
            "content": response.get("response", ""),
            "role": "assistant"
        }
    
    async def stream_completion(self,
                              messages: List[Dict[str, Any]],
                              tools: Optional[List[Dict]] = None,
                              temperature: float = 0.7,
                              max_tokens: Optional[int] = None,
                              **kwargs) -> AsyncIterator[Dict]:
        """Stream completion tokens."""
        prompt = self._messages_to_prompt(messages)
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
            }
        }
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        payload["options"].update(kwargs)
        
        async for chunk in self._stream_completion(payload):
            yield chunk
    
    def get_max_tokens(self) -> int:
        """Get maximum tokens for model."""
        # Most Ollama models support 2048-8192 tokens
        return 4096
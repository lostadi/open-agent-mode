"""Google Gemini provider implementation."""

import os
import json
from typing import Dict, List, Any, Optional, AsyncIterator
import google.generativeai as genai
import logging

from .base import LLMProvider


logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    """Google Gemini API provider."""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "gemini-1.5-pro"):
        """Initialize Gemini provider.
        
        Args:
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
            model: Model to use
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.model = model
        
        if not self.api_key:
            raise ValueError("Google API key is required")
        
        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(model)
    
    async def get_completion(self,
                            messages: List[Dict[str, Any]],
                            tools: Optional[List[Dict]] = None,
                            temperature: float = 0.7,
                            max_tokens: Optional[int] = None,
                            stream: bool = False,
                            **kwargs) -> Any:
        """Get completion from Gemini.
        
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
            # Convert OpenAI format to Gemini format
            gemini_messages = self._convert_messages(messages)
            
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens or 8192,
            }
            generation_config.update(kwargs)
            
            # Convert tools if provided
            gemini_tools = None
            if tools:
                gemini_tools = self._convert_tools(tools)
            
            if stream:
                return self._stream_completion(gemini_messages, generation_config, gemini_tools)
            else:
                if gemini_tools:
                    response = self.client.generate_content(
                        gemini_messages,
                        generation_config=generation_config,
                        tools=gemini_tools
                    )
                else:
                    response = self.client.generate_content(
                        gemini_messages,
                        generation_config=generation_config
                    )
                return self._format_response(response)
        
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise
    
    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[Dict]:
        """Convert OpenAI format messages to Gemini format."""
        gemini_messages = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            # Map roles
            if role == "system":
                # Gemini doesn't have system role, prepend to first user message
                gemini_messages.insert(0, {
                    "role": "user",
                    "parts": [{"text": f"System: {content}"}]
                })
            elif role == "assistant":
                gemini_messages.append({
                    "role": "model",
                    "parts": [{"text": content}]
                })
            elif role == "user":
                gemini_messages.append({
                    "role": "user",
                    "parts": [{"text": content}]
                })
            elif role == "tool":
                # Handle tool results
                gemini_messages.append({
                    "role": "function",
                    "parts": [{"text": content}]
                })
        
        return gemini_messages
    
    def _convert_tools(self, tools: List[Dict]) -> List:
        """Convert OpenAI tool format to Gemini format."""
        from google.generativeai.types import FunctionDeclaration, Tool
        
        gemini_functions = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                
                # Convert parameters
                parameters = func.get("parameters", {})
                
                function_decl = FunctionDeclaration(
                    name=func["name"],
                    description=func.get("description", ""),
                    parameters=parameters
                )
                gemini_functions.append(function_decl)
        
        return [Tool(function_declarations=gemini_functions)] if gemini_functions else None
    
    async def _stream_completion(self, messages: List, config: Dict, tools: Optional[List]) -> AsyncIterator[Dict]:
        """Stream completion from Gemini."""
        try:
            if tools:
                response = self.client.generate_content(
                    messages,
                    generation_config=config,
                    tools=tools,
                    stream=True
                )
            else:
                response = self.client.generate_content(
                    messages,
                    generation_config=config,
                    stream=True
                )
            
            for chunk in response:
                result = {}
                
                if chunk.text:
                    result["content"] = chunk.text
                
                # Handle function calls
                if hasattr(chunk, 'parts'):
                    for part in chunk.parts:
                        if hasattr(part, 'function_call'):
                            if "tool_calls" not in result:
                                result["tool_calls"] = []
                            
                            result["tool_calls"].append({
                                "id": f"call_{part.function_call.name}",
                                "type": "function",
                                "function": {
                                    "name": part.function_call.name,
                                    "arguments": json.dumps(dict(part.function_call.args))
                                }
                            })
                
                if result:
                    yield result
        
        except Exception as e:
            logger.error(f"Gemini streaming error: {e}")
            raise
    
    def _format_response(self, response) -> Dict[str, Any]:
        """Format Gemini response to standard format."""
        result = {
            "content": "",
            "role": "assistant"
        }
        
        if response.text:
            result["content"] = response.text
        
        # Handle function calls
        if hasattr(response, 'parts'):
            for part in response.parts:
                if hasattr(part, 'function_call'):
                    if "tool_calls" not in result:
                        result["tool_calls"] = []
                    
                    result["tool_calls"].append({
                        "id": f"call_{part.function_call.name}",
                        "type": "function",
                        "function": {
                            "name": part.function_call.name,
                            "arguments": json.dumps(dict(part.function_call.args))
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
        gemini_messages = self._convert_messages(messages)
        
        config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens or 8192,
        }
        config.update(kwargs)
        
        gemini_tools = None
        if tools:
            gemini_tools = self._convert_tools(tools)
        
        async for chunk in self._stream_completion(gemini_messages, config, gemini_tools):
            yield chunk
    
    def get_max_tokens(self) -> int:
        """Get maximum tokens for model."""
        model_limits = {
            "gemini-1.5-pro": 2000000,
            "gemini-1.5-flash": 1000000,
            "gemini-pro": 32000,
        }
        
        for model_name, limit in model_limits.items():
            if model_name in self.model:
                return limit
        
        return 32000  # Default
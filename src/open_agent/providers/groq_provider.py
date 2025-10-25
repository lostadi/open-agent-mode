"""Groq provider for ultra-fast inference."""

import os
from typing import Optional
from groq import AsyncGroq

from .openai_provider import OpenAIProvider


class GroqProvider(OpenAIProvider):
    """Groq API provider (OpenAI-compatible)."""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "llama-3.1-70b-versatile"):
        """Initialize Groq provider.
        
        Args:
            api_key: Groq API key (defaults to GROQ_API_KEY env var)
            model: Model to use
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model
        
        if not self.api_key:
            raise ValueError("Groq API key is required")
        
        self.client = AsyncGroq(api_key=self.api_key)
    
    def get_max_tokens(self) -> int:
        """Get maximum tokens for model."""
        model_limits = {
            "llama-3.1-405b": 131072,
            "llama-3.1-70b": 131072,
            "llama-3.1-8b": 131072,
            "mixtral-8x7b": 32768,
            "gemma-7b": 8192,
        }
        
        for model_name, limit in model_limits.items():
            if model_name in self.model:
                return limit
        
        return 8192  # Default
"""Base LLM provider interface."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, AsyncIterator


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def get_completion(self,
                            messages: List[Dict[str, Any]],
                            tools: Optional[List[Dict]] = None,
                            temperature: float = 0.7,
                            max_tokens: Optional[int] = None,
                            stream: bool = False,
                            **kwargs) -> Any:
        """Get completion from the LLM.
        
        Args:
            messages: Conversation messages in OpenAI format
            tools: Available tools/functions
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Provider-specific parameters
        
        Returns:
            Completion response or stream
        """
        pass
    
    @abstractmethod
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
            **kwargs: Provider-specific parameters
        
        Yields:
            Response chunks
        """
        pass
    
    def get_token_count(self, text: str) -> int:
        """Get token count for text.
        
        Args:
            text: Text to count tokens for
        
        Returns:
            Token count (approximate if exact count not available)
        """
        # Default implementation - rough estimate
        return len(text) // 4
    
    def get_max_tokens(self) -> int:
        """Get maximum tokens for model.
        
        Returns:
            Maximum token limit
        """
        return 4096  # Default conservative limit

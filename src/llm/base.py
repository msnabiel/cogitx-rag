"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from typing import List, Optional
from loguru import logger


class BaseLLM(ABC):
    """Abstract LLM provider interface."""

    def __init__(self, model_name: str):
        """
        Initialize LLM provider.

        Args:
            model_name: Name of the model
        """
        self.model_name = model_name
        logger.info(f"Initialized {self.__class__.__name__} with model: {model_name}")

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate text completion.

        Args:
            prompt: User prompt
            system_prompt: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens (None = no limit)

        Returns:
            Generated text
        """
        pass

    @abstractmethod
    async def chat(
        self,
        messages: List[dict],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Chat completion.

        Args:
            messages: List of message dictionaries
            temperature: Sampling temperature
            max_tokens: Maximum tokens (None = no limit)

        Returns:
            Assistant response
        """
        pass

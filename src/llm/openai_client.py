"""OpenAI LLM client."""

from typing import List, Optional
from openai import AsyncOpenAI
from loguru import logger
from llm.base import BaseLLM
from core.exceptions import LLMError
from config.settings import settings


class OpenAIClient(BaseLLM):
    """OpenAI LLM client."""

    def __init__(self, api_key: str = None, model_name: str = None):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key
            model_name: Model name
        """
        api_key = api_key or settings.llm.openai_api_key
        model_name = model_name or settings.llm.openai_model

        if not api_key:
            raise LLMError("OpenAI API key not provided")

        super().__init__(model_name=model_name)
        self.client = AsyncOpenAI(api_key=api_key)

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate completion."""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            kwargs = {"model": self.model_name, "messages": messages, "temperature": temperature}
            if max_tokens:
                kwargs["max_tokens"] = max_tokens

            response = await self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise LLMError(f"Generation failed: {str(e)}")

    async def chat(
        self,
        messages: List[dict],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Chat completion."""
        try:
            kwargs = {"model": self.model_name, "messages": messages, "temperature": temperature}
            if max_tokens:
                kwargs["max_tokens"] = max_tokens

            response = await self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI chat error: {e}")
            raise LLMError(f"Chat failed: {str(e)}")

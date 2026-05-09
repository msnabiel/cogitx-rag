"""Google Gemini LLM client."""

from typing import List, Optional
import google.generativeai as genai
from loguru import logger
from src.llm.base import BaseLLM
from src.core.types_and_exception import LLMError
from src.config.settings import settings


class GeminiClient(BaseLLM):
    """Google Gemini LLM client."""

    def __init__(self, api_key: str = None, model_name: str = None):
        """
        Initialize Gemini client.

        Args:
            api_key: Gemini API key
            model_name: Model name
        """
        api_key = api_key or settings.llm.gemini_api_key
        model_name = model_name or settings.llm.gemini_model

        if not api_key:
            raise LLMError("Gemini API key not provided")

        super().__init__(model_name=model_name)

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate completion."""
        try:
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            config_args = {"temperature": temperature}
            if max_tokens:
                config_args["max_output_tokens"] = max_tokens

            response = self.model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(**config_args),
            )
            return response.text

        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            raise LLMError(f"Generation failed: {str(e)}")

    async def chat(
        self,
        messages: List[dict],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Chat completion."""
        try:
            chat = self.model.start_chat(history=[])

            for msg in messages[:-1]:
                chat.send_message(msg["content"])

            config_args = {"temperature": temperature}
            if max_tokens:
                config_args["max_output_tokens"] = max_tokens

            response = chat.send_message(
                messages[-1]["content"],
                generation_config=genai.types.GenerationConfig(**config_args),
            )
            return response.text

        except Exception as e:
            logger.error(f"Gemini chat error: {e}")
            raise LLMError(f"Chat failed: {str(e)}")

"""Query processing with LLM"""

import logging
import asyncio
from src.config.settings import settings

logger = logging.getLogger(__name__)


class ProcessQuery:
    """Handles query processing with LLM"""

    def __init__(self, search_methods, llm_client, generation_config):
        self.search_methods = search_methods
        self.client = llm_client
        self.generation_config = generation_config

    async def process(self, query: str) -> str:
        """Process query using ensemble search and LLM"""
        from src.utils.text_cleaner import fuzzy_matching, TextCleaner, CleaningOptions
        from src.utils.prompt_loader import load_prompt

        try:
            # Retrieve chunks
            search_results = self.search_methods.ensemble_search(query, top_k=16)
            relevant_chunks = [result.chunk for result in search_results]

            # Deduplicate
            chunk_texts = [chunk.text for chunk in relevant_chunks]
            unique_texts = fuzzy_matching(chunk_texts, min_ratio=85)
            deduplicated_chunks = []
            for text in unique_texts:
                for chunk in relevant_chunks:
                    if chunk.text == text:
                        deduplicated_chunks.append(chunk)
                        break

            # Build context
            context_parts = [
                f"[Chunk {i+1}]: {chunk.text}"
                for i, chunk in enumerate(deduplicated_chunks)
            ]
            cleaner = TextCleaner(CleaningOptions())
            context = cleaner.clean_text("\n\n".join(context_parts))

            # Load prompt
            try:
                prompt_template = load_prompt("prompts/phase1_prompt.txt")
            except FileNotFoundError:
                logger.warning("Prompt file not found, using default")
                prompt_template = """Based on the following context, answer the question.

Context: {context}

Question: {query}

Answer:"""

            prompt = prompt_template.format(context=context, query=query)
            logger.info(f"Processing query with LLM")

            # Call LLM
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=settings.llm.gemini_model,
                config=self.generation_config,
                contents=prompt
            )

            if not hasattr(response, "text"):
                logger.warning("LLM response missing text")
                return "No response generated"

            answer = response.text.strip()
            logger.info(f"Generated answer")
            return answer

        except Exception as e:
            logger.error(f"Error in process: {e}")
            return f"Error processing query: {e}"


__all__ = ['ProcessQuery']

"""Query processing with LLM"""

import logging
import asyncio
from src.config.settings import settings
from src.llm.prompt_templates import PromptTemplates
from src.core.models import ContextWindow

logger = logging.getLogger(__name__)


class ProcessQuery:
    """Handles query processing with LLM"""

    def __init__(self, search_methods, llm_client, generation_config, memory_manager=None, system_prompt=""):
        self.search_methods = search_methods
        self.client = llm_client
        self.generation_config = generation_config
        self.memory_manager = memory_manager
        self.system_prompt = system_prompt

    async def process(self, query: str, session_id: str = None) -> str:
        """Process query using ensemble search and LLM"""
        from src.utils.text_cleaner import fuzzy_matching

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

            retrieved_contexts = [f"[Chunk {i+1}]: {chunk.text}" for i, chunk in enumerate(deduplicated_chunks)]
            memory_context = self.memory_manager.get_context(session_id) if self.memory_manager and session_id else ""
            context_window = ContextWindow(
                query=query,
                retrieved_contexts=retrieved_contexts,
                graph_context=None,
                memory_context=memory_context or None,
                system_prompt=self.system_prompt or "You are a helpful assistant.",
                total_tokens=0,
            )
            prompt = PromptTemplates.build_rag_prompt(context_window)
            logger.info(
                "Gemini prompt for session %s:\n%s",
                session_id or "no-session",
                prompt[:4000],
            )
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
            if self.memory_manager and session_id:
                self.memory_manager.append_turn(session_id, "user", query)
                self.memory_manager.append_turn(session_id, "assistant", answer)
            logger.info(f"Generated answer")
            return answer

        except Exception as e:
            logger.error(f"Error in process: {e}")
            return f"Error processing query: {e}"


__all__ = ['ProcessQuery']

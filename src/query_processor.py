"""Query processing with LLM"""

import logging
import re
import json
from types import SimpleNamespace
from src.utils.prompt_loader import load_prompt

logger = logging.getLogger(__name__)


class ProcessQuery:
    """Handles query processing with LLM"""

    def __init__(self, search_methods, llm_client, generation_config, memory_manager=None, system_prompt=""):
        self.search_methods = search_methods
        self.client = llm_client
        self.generation_config = generation_config
        self.memory_manager = memory_manager
        self.system_prompt = system_prompt
        self.rag_prompt_template = load_prompt("prompts/rag_prompt.txt")

    def _clean_answer(self, answer: str) -> str:
        """Remove prompt artifacts from the model response."""
        cleaned = answer.strip()
        cleaned = re.sub(r"^\s*Answer:\s*", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def _parse_json_response(self, response: str):
        """Parse the model response as strict JSON, with a small fallback for fenced output."""
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        payload = json.loads(cleaned)
        answer = self._clean_answer(str(payload.get("answer", "")))
        confidence = float(payload.get("confidence", 0.0))
        citations = payload.get("citations", [])
        return answer, confidence, citations

    def _format_source_label(self, chunk, index: int) -> str:
        """Build a human-readable source label from chunk metadata."""
        metadata = getattr(chunk, "metadata", {}) or {}
        filename = metadata.get("filename") or metadata.get("source") or metadata.get("title") or "Document"
        page = metadata.get("page") or metadata.get("page_number")
        if page is not None:
            return f"[{index}] {filename} (page {page})"
        return f"[{index}] {filename}"

    def _build_citations(self, chunks):
        """Build citation records from retrieved chunks."""
        citations = []
        for i, chunk in enumerate(chunks, 1):
            citations.append({
                "citation": f"[{i}]",
                "label": self._format_source_label(chunk, i),
                "content": chunk.text,
                "chunk_id": chunk.chunk_id,
                "metadata": chunk.metadata,
            })
        return citations

    def _render_prompt(self, memory_context: str, retrieved_contexts: str, query: str) -> str:
        """Render the prompt template while preserving literal JSON braces."""
        template = self.rag_prompt_template.replace("{", "{{").replace("}", "}}")
        template = template.replace("{{memory_context}}", "{memory_context}")
        template = template.replace("{{retrieved_contexts}}", "{retrieved_contexts}")
        template = template.replace("{{query}}", "{query}")
        return template.format(
            memory_context=memory_context or "None",
            retrieved_contexts=retrieved_contexts or "None",
            query=query,
        )

    async def process(self, query: str, session_id: str = None):
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

            retrieved_contexts = [f"[{i+1}] {chunk.text}" for i, chunk in enumerate(deduplicated_chunks)]
            memory_context = self.memory_manager.get_context(session_id) if self.memory_manager and session_id else ""
            prompt = self._render_prompt(
                memory_context=memory_context or "None",
                retrieved_contexts="\n".join(retrieved_contexts) or "None",
                query=query,
            )
            logger.info(
                "%s prompt for session %s:\n%s",
                self.client.__class__.__name__,
                session_id or "no-session",
                prompt[:4000],
            )
            logger.info(f"Processing query with LLM")

            # Call LLM
            response = await self.client.generate(
                prompt=prompt,
                temperature=getattr(self.generation_config, "temperature", 0.7),
                max_tokens=getattr(self.generation_config, "max_output_tokens", None),
            )

            if not response:
                logger.warning("LLM response missing text")
                return "No response generated"

            answer, confidence, _ = self._parse_json_response(response)
            citations = self._build_citations(deduplicated_chunks)
            if confidence <= 0.0 and search_results:
                top_score = max(result.combined_score for result in search_results)
                confidence = min(1.0, round(top_score * 5, 3))
            if self.memory_manager and session_id:
                self.memory_manager.append_turn(session_id, "user", query)
                self.memory_manager.append_turn(session_id, "assistant", answer)
            logger.info(f"Generated answer")
            return SimpleNamespace(
                answer=answer,
                citations=citations,
                confidence=confidence,
                sources=deduplicated_chunks,
            )

        except Exception as e:
            logger.error(f"Error in process: {e}")
            return SimpleNamespace(
                answer=f"Error processing query: {e}",
                citations=[],
                confidence=0.0,
                sources=[],
            )


__all__ = ['ProcessQuery']

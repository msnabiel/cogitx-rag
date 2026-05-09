"""Prompt templates for RAG system."""

from typing import List, Optional
from src.core.models import ContextWindow


class PromptTemplates:
    """RAG prompt templates."""

    @staticmethod
    def build_rag_prompt(context_window: ContextWindow) -> str:
        """
        Build RAG prompt from context window.

        Args:
            context_window: Context window

        Returns:
            Formatted prompt
        """
        prompt_parts = []

        # Add system prompt
        prompt_parts.append(context_window.system_prompt)

        # Add memory context if available
        if context_window.memory_context:
            prompt_parts.append(f"\n## User Context\n{context_window.memory_context}")

        # Add graph context if available
        if context_window.graph_context:
            prompt_parts.append(f"\n## Knowledge Graph Context\n{context_window.graph_context}")

        # Add retrieved contexts
        if context_window.retrieved_contexts:
            prompt_parts.append("\n## Retrieved Information")
            for i, context in enumerate(context_window.retrieved_contexts, 1):
                prompt_parts.append(f"\n### Source {i}\n{context}")

        # Add query
        prompt_parts.append(f"\n## User Question\n{context_window.query}")

        prompt_parts.append("\n## Answer")

        return "\n".join(prompt_parts)

    @staticmethod
    def build_chat_messages(
        context_window: ContextWindow,
        chat_history: Optional[List[dict]] = None,
    ) -> List[dict]:
        """
        Build chat messages format.

        Args:
            context_window: Context window
            chat_history: Previous chat history

        Returns:
            List of message dictionaries
        """
        messages = []

        # System message
        system_content = [context_window.system_prompt]

        if context_window.memory_context:
            system_content.append(f"\nUser Context:\n{context_window.memory_context}")

        messages.append({
            "role": "system",
            "content": "\n".join(system_content),
        })

        # Add chat history
        if chat_history:
            messages.extend(chat_history)

        # Build user message with context
        user_content_parts = []

        if context_window.graph_context:
            user_content_parts.append(f"Knowledge Graph Context:\n{context_window.graph_context}\n")

        if context_window.retrieved_contexts:
            user_content_parts.append("Retrieved Information:")
            for i, context in enumerate(context_window.retrieved_contexts, 1):
                user_content_parts.append(f"\nSource {i}:\n{context}")
            user_content_parts.append("")

        user_content_parts.append(f"Question: {context_window.query}")

        messages.append({
            "role": "user",
            "content": "\n".join(user_content_parts),
        })

        return messages

    @staticmethod
    def format_sources(contexts: List[str]) -> str:
        """
        Format sources for citation.

        Args:
            contexts: Retrieved contexts

        Returns:
            Formatted sources string
        """
        if not contexts:
            return ""

        sources = [f"[{i+1}] {ctx[:200]}..." for i, ctx in enumerate(contexts)]
        return "\n\nSources:\n" + "\n".join(sources)

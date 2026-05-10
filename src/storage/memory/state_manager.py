"""Persistence for retrieval state and conversational memory."""

import os
import pickle
from typing import Any, Optional

import faiss


class RAGStateManager:
    """Persist and restore RAG retrieval state."""

    def __init__(self, state_dir: str, logger):
        self.state_dir = state_dir
        self.logger = logger
        os.makedirs(state_dir, exist_ok=True)
        self.faiss_file = os.path.join(state_dir, "faiss.index")
        self.bm25_file = os.path.join(state_dir, "bm25.pkl")
        self.chunks_file = os.path.join(state_dir, "chunks.pkl")
        self.bge_embeddings_file = os.path.join(state_dir, "bge_embeddings.pkl")
        self.all_mini_embeddings_file = os.path.join(state_dir, "all_mini_embeddings.pkl")
        self.combined_embeddings_file = os.path.join(state_dir, "combined_embeddings.pkl")

    def save(self, *, faiss_index, bm25, chunks, bge_embeddings, all_mini_embeddings, combined_embeddings) -> None:
        """Persist retrieval state to disk."""
        if faiss_index is None or bm25 is None or not chunks:
            return

        faiss.write_index(faiss_index, self.faiss_file)
        with open(self.bm25_file, "wb") as f:
            pickle.dump(bm25, f)
        with open(self.chunks_file, "wb") as f:
            pickle.dump(chunks, f)
        with open(self.bge_embeddings_file, "wb") as f:
            pickle.dump(bge_embeddings, f)
        with open(self.all_mini_embeddings_file, "wb") as f:
            pickle.dump(all_mini_embeddings, f)
        with open(self.combined_embeddings_file, "wb") as f:
            pickle.dump(combined_embeddings, f)

    def load(self) -> Optional[dict[str, Any]]:
        """Load retrieval state from disk if all files exist."""
        required = [
            self.faiss_file,
            self.bm25_file,
            self.chunks_file,
            self.bge_embeddings_file,
            self.all_mini_embeddings_file,
            self.combined_embeddings_file,
        ]
        if not all(os.path.exists(path) for path in required):
            return None

        with open(self.bm25_file, "rb") as f:
            bm25 = pickle.load(f)
        with open(self.chunks_file, "rb") as f:
            chunks = pickle.load(f)
        with open(self.bge_embeddings_file, "rb") as f:
            bge_embeddings = pickle.load(f)
        with open(self.all_mini_embeddings_file, "rb") as f:
            all_mini_embeddings = pickle.load(f)
        with open(self.combined_embeddings_file, "rb") as f:
            combined_embeddings = pickle.load(f)

        return {
            "faiss_index": faiss.read_index(self.faiss_file),
            "bm25": bm25,
            "chunks": chunks,
            "bge_embeddings": bge_embeddings,
            "all_mini_embeddings": all_mini_embeddings,
            "combined_embeddings": combined_embeddings,
        }


class ConversationMemoryManager:
    """Sliding-window conversation memory with rolling summaries."""

    def __init__(self, cache_dir: str, logger, window_size: int = 6, overflow_threshold: int = 10, llm_client=None):
        self.cache_dir = cache_dir
        self.logger = logger
        self.window_size = window_size
        self.overflow_threshold = overflow_threshold
        self.llm_client = llm_client
        self.conversations_dir = os.path.join(cache_dir, "conversations")
        os.makedirs(self.conversations_dir, exist_ok=True)

    def _path(self, session_id: str) -> str:
        return os.path.join(self.conversations_dir, f"{session_id}.pkl")

    def _load_state(self, session_id: str) -> dict:
        path = self._path(session_id)
        if not os.path.exists(path):
            return {"turns": [], "summary": "", "overflow": []}
        with open(path, "rb") as f:
            state = pickle.load(f)
            state.setdefault("turns", [])
            state.setdefault("summary", "")
            state.setdefault("overflow", [])
            return state

    def _save_state(self, session_id: str, state: dict) -> None:
        with open(self._path(session_id), "wb") as f:
            pickle.dump(state, f)

    def get_context(self, session_id: str) -> str:
        state = self._load_state(session_id)
        turns = state.get("turns", [])
        summary = state.get("summary", "")

        recent = turns[-self.window_size :]
        parts = []
        if summary:
            parts.append(f"Conversation summary:\n{summary}")
        overflow = state.get("overflow", [])
        if overflow:
            parts.append("Overflow buffer:")
            for turn in overflow:
                parts.append(f"{turn['role']}: {turn['content']}")
        if recent:
            parts.append("Recent turns:")
            for turn in recent:
                parts.append(f"{turn['role']}: {turn['content']}")
        return "\n\n".join(parts)

    async def append_turn(self, session_id: str, role: str, content: str) -> str:
        state = self._load_state(session_id)
        turns = state.get("turns", [])
        turns.append({"role": role, "content": content})

        summary = state.get("summary", "")
        overflow_buffer = state.get("overflow", [])

        if len(turns) > self.window_size:
            overflow_buffer.extend(turns[:-self.window_size])
            turns = turns[-self.window_size :]

        if len(overflow_buffer) >= self.overflow_threshold:
            summary = await self._summarize_turns(summary, overflow_buffer)
            overflow_buffer = []

        state["turns"] = turns
        state["summary"] = summary
        state["overflow"] = overflow_buffer
        self._save_state(session_id, state)
        return self.get_context(session_id)

    async def _summarize_turns(self, existing_summary: str, turns: list[dict]) -> str:
        transcript = "\n".join(f"{turn['role']}: {turn['content']}" for turn in turns)
        if self.llm_client is None:
            snippets = []
            if existing_summary:
                snippets.append(existing_summary.strip())
            snippets.append(transcript[:1200])
            return " | ".join(snippets)

        prompt = (
            "Summarize the following conversation turns into a compact long-term memory. "
            "Keep concrete entities, user goals, constraints, tools, and decisions. "
            "Return plain text only.\n\n"
            f"Existing summary:\n{existing_summary or 'None'}\n\n"
            f"Turns to summarize:\n{transcript}"
        )
        try:
            summary = await self.llm_client.generate(prompt=prompt, temperature=0.2, max_tokens=200)
            return summary.strip()
        except Exception as e:
            self.logger.warning(f"Memory summarization failed, falling back to transcript compression: {e}")
            snippets = []
            if existing_summary:
                snippets.append(existing_summary.strip())
            snippets.append(transcript[:1200])
            return " | ".join(snippets)

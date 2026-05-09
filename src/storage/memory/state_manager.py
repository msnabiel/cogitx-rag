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

    def __init__(self, cache_dir: str, logger, window_size: int = 6):
        self.cache_dir = cache_dir
        self.logger = logger
        self.window_size = window_size
        self.conversations_dir = os.path.join(cache_dir, "conversations")
        os.makedirs(self.conversations_dir, exist_ok=True)

    def _path(self, session_id: str) -> str:
        return os.path.join(self.conversations_dir, f"{session_id}.pkl")

    def _load_state(self, session_id: str) -> dict:
        path = self._path(session_id)
        if not os.path.exists(path):
            return {"turns": [], "summary": ""}
        with open(path, "rb") as f:
            return pickle.load(f)

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
        if recent:
            parts.append("Recent turns:")
            for turn in recent:
                parts.append(f"{turn['role']}: {turn['content']}")
        return "\n\n".join(parts)

    def append_turn(self, session_id: str, role: str, content: str) -> str:
        state = self._load_state(session_id)
        turns = state.get("turns", [])
        turns.append({"role": role, "content": content})

        summary = state.get("summary", "")
        overflow = max(0, len(turns) - self.window_size)
        if overflow > 0:
            old_turns = turns[:overflow]
            summary = self._summarize_turns(summary, old_turns)
            turns = turns[overflow:]

        state["turns"] = turns
        state["summary"] = summary
        self._save_state(session_id, state)
        return self.get_context(session_id)

    def _summarize_turns(self, existing_summary: str, turns: list[dict]) -> str:
        snippets = []
        if existing_summary:
            snippets.append(existing_summary.strip())
        for turn in turns:
            snippets.append(f"{turn['role']}: {turn['content'][:200]}")
        combined = " | ".join(snippets)
        if len(combined) > 1200:
            combined = combined[:1200].rsplit(" ", 1)[0]
        return combined

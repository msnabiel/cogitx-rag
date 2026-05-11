# CogitX-RAG

FastAPI-based RAG system with document ingestion, conversational memory, Slack/Telegram integrations, configurable embeddings, and pluggable vector stores.

## Architecture

![CogitX-RAG Architecture](assets/architecture.png)

## Demo

[![Watch Demo](/assets/demo-thumbnail.png)](https://drive.google.com/file/d/1EjqoRS3pI4mJeXKQSxurDXGZ_lO1P3Bl/view?usp=sharing)

## System Overview

| Component | Details |
|---|---|
| Backend | FastAPI |
| Retrieval | Chunk-based semantic retrieval |
| Embeddings | Local, OpenAI, Gemini |
| Vector Stores | FAISS, Pinecone |
| Memory | Recent window + overflow summarization |
| Bots | Slack, Telegram |
| OCR Support | Tesseract + LibreOffice |
| Runtime | Uvicorn |

---

## Features

| Feature | Status | Notes |
|---|---|---|
| Document upload | Working | API upload + ingestion pipeline |
| Chunking & embeddings | Working | Configurable embedding providers |
| Retrieval & generation | Working | Retrieval + LLM response flow |
| Confidence scoring | Working | Returned in query response |
| Slack bot | Working | Thread-based session memory |
| Telegram bot | Partial | Present but lightly tested |
| Session memory | Working | Window + summarization |
| Pinecone support | Partial | Requires validation |
| Graph retrieval | Pending | Modules exist, not integrated |
| Semantic memory | Pending | Runtime wiring incomplete |
| Structured memory | Pending | Runtime wiring incomplete |

---

## Embedding Modes

| Mode | Description |
|---|---|
| `local_single` | Uses one local embedding model |
| `local_dual` | Concatenates two local models |
| `openai` | OpenAI embeddings |
| `gemini` | Gemini embeddings |

### Default Local Models

| Model | Dimension |
|---|---|
| `BAAI/bge-small-en-v1.5` | 384 |
| `all-MiniLM-L6-v2` | 384 |

Combined vector size in `local_dual` mode: **768**

---

## Memory System

| Behavior | Description |
|---|---|
| Recent window | Keeps latest conversation turns |
| Overflow storage | Moves older turns out of active window |
| Summarization | Uses LLM to summarize overflow |
| Prompt assembly | Injects summary + recent history |

Active runtime path:

```text
src/storage/memory/state_manager.py
```

---

## Configuration

| File | Purpose |
|---|---|
| `settings.yaml` | Main application config |
| `.env` | Secrets and API keys |
| `src/config/settings.py` | Config loader |

### Important Config Keys

| Key | Purpose |
|---|---|
| `llm.default_llm_provider` | Active LLM backend |
| `embeddings.embedding_provider` | Embedding provider |
| `vector_store.vector_store_type` | FAISS / Pinecone |
| `memory.conversation_window_size` | Recent history size |
| `memory.overflow_summary_threshold` | Summarization trigger |
| `slack.slack_enabled` | Slack startup toggle |
| `telegram.telegram_enabled` | Telegram startup toggle |

---

## Prompt Files

| File | Purpose |
|---|---|
| `prompts/system_prompt.txt` | System instructions |
| `prompts/rag_prompt.txt` | Main RAG prompt |

---

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/v1/upload` | Upload document |
| POST | `/api/v1/upload-query` | Upload + query |
| POST | `/api/v1/ingest` | Ingest documents |
| POST | `/api/v1/query` | Query RAG pipeline |
| POST | `/api/v1/search` | Semantic search |

### Query Response

```json
{
  "answer": "...",
  "confidence": 0.0
}
```

---

## Vector Stores

| Store | Type |
|---|---|
| FAISS | Local vector database |
| Pinecone | Managed cloud vector database |

---

## Running Locally

```bash
python3 main.py
```

Development reload is enabled when:

```text
environment != production
```

---

## Current Limitations

| Area | Issue |
|---|---|
| Pinecone | Needs ingest/query validation |
| Telegram | Limited runtime testing |
| Citations | PDF metadata can be noisy |
| Graph retrieval | Not wired into runtime |
| Semantic memory | Exists but inactive |
| Structured memory | Exists but inactive |

---

## Project Structure

```text
src/
├── api/
├── bots/
├── config/
├── ingestion/
├── query/
├── storage/
│   ├── embeddings/
│   ├── memory/
│   └── vectorstores/
├── utils/
└── prompts/
```
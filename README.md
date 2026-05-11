# CogitX-RAG

CogitX-RAG is a FastAPI RAG system with document upload, Slack/Telegram bots, session memory, and pluggable embeddings/vector stores.

## Working

| Area | Status |
|---|---|
| Upload, ingest, query, citations | Working |
| Slack bot with thread session memory | Working |
| Telegram bot, Pinecone path, persistent retrieval state | Present and partially wired |

## Current Behavior

- Documents are uploaded, extracted, chunked, embedded, and indexed.
- Query responses include `answer`, `citations`, and `confidence`.
- Local embedding modes support `local_single` and `local_dual`.
- `local_dual` concatenates `local_embedding_model_1` and `local_embedding_model_2`.
- Conversational memory uses `ConversationMemoryManager` with a recent window plus overflow summary.
- Prompt templates are file-backed and loaded with `src/utils/prompt_loader.py`.

## Configuration

- Non-secret config lives in [`settings.yaml`](settings.yaml).
- Secrets live in [`.env`](.env).
- [`src/config/settings.py`](src/config/settings.py) loads YAML plus secrets.

## Prompt Files

- [`prompts/system_prompt.txt`](prompts/system_prompt.txt)
- [`prompts/rag_prompt.txt`](prompts/rag_prompt.txt)

## API

- `POST /api/v1/upload`
- `POST /api/v1/upload-query`
- `POST /api/v1/ingest`
- `POST /api/v1/query`
- `POST /api/v1/search`

## TODO

| Item | Notes |
|---|---|
| Graph retrieval | Not integrated into the current runtime path |
| Telegram bot | Present, not fully tested end-to-end |
| Citations metadata | Page/line numbering can be noisy and truncated |
| Pinecone | Needs full verification on ingest/query symmetry |
| SemanticMemory / StructuredMemory | Modules exist, wiring and tests still pending |
| Dockerfile | Needs runtime validation against current branch changes |

## Notes

- `config.py` at the repo root has been removed because it is not used by `main.py`.
- The active prompt assembly path now uses `prompts/rag_prompt.txt` only.

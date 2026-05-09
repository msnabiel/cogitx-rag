# CogitX-RAG Quick Start

## Setup

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Configure environment**
```bash
cp .env.example .env.local
# Edit .env.local and add your GEMINI_API_KEY_PAID
```

3. **Create directories**
```bash
mkdir -p logs uploaded_docs prompts
```

4. **Add system prompt** (optional)
Create `prompts/system_prompt.txt` with your system instructions, or leave empty for default.

## Run

```bash
python main.py
```

Server starts at `http://0.0.0.0:8000`

## API Endpoints

### 1. Upload file
```bash
curl -X POST "http://localhost:8000/api/v1/upload" \
  -F "file=@document.pdf"
```
Returns: `{"url": "file://.../document.pdf", ...}`

### 2. Ingest documents
```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -H "Content-Type: application/json" \
  -d '{"file_paths": ["file:///path/to/doc1.pdf", "file:///path/to/doc2.pdf"]}'
```

### 3. Search documents
```bash
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?", "top_k": 10}'
```

### 4. Query with LLM
```bash
curl -X POST "http://localhost:8000/api/v1/query?query=Explain%20RAG%20systems"
```

### 5. Cache stats
```bash
curl "http://localhost:8000/api/v1/cache/stats"
```

### 6. Clear cache
```bash
curl -X POST "http://localhost:8000/api/v1/cache/clear"
```

## Flow

1. **Upload** → Get file URL
2. **Ingest** → Process files, build indices (FAISS + BM25)
3. **Search** → Semantic + lexical ensemble search
4. **Query** → Retrieve + LLM answer generation

## Settings

Edit `src/config/settings.py` for:
- Upload/cache directories
- Chunk size, overlap
- Embedding batch size
- Similarity thresholds
- GPU/CPU workers

## GPU Support

Auto-detects CUDA. Models moved to GPU if available.

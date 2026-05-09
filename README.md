# CogitX-RAG 🧠

A production-ready **Graph-based RAG (Retrieval-Augmented Generation) system** with dual-path retrieval combining vector search, BM25 keyword matching, and knowledge graph traversal.

## 🎯 Features

### Core Capabilities
- **Dual-Path Retrieval**: Parallel execution of semantic (vector), lexical (BM25), and structural (graph) search
- **Knowledge Graph Integration**: Neo4j-based entity and relationship extraction with multi-hop traversal
- **Multiple LLM Support**: Switchable between OpenAI and Google Gemini
- **Flexible Vector Stores**: Runtime switching between Pinecone (cloud) and FAISS (local)
- **Advanced Reranking**: Reciprocal Rank Fusion (RRF) and cross-encoder reranking
- **Memory Layers**: Short-term (Redis cache), long-term (vector DB), and structured (graph) memory
- **Context Compression**: Intelligent context windowing to fit LLM token limits
- **REST API**: Production-ready FastAPI with health checks and monitoring

### Architecture Highlights
- **Modular Design**: Clean separation of concerns with pluggable components
- **Async First**: Full async/await support for high concurrency
- **Type Safety**: Pydantic models throughout for validation
- **Observable**: Comprehensive logging with Loguru
- **Scalable**: Designed for evolution to graph-first and agent-based strategies

## 📁 Project Structure

```
cogitx-rag/
├── config/              # Configuration and settings
├── core/                # Core models, types, exceptions
├── embeddings/          # Embedding providers (OpenAI, Gemini)
├── vector_stores/       # Vector store implementations (Pinecone, FAISS)
├── graph/               # Neo4j client, entity extraction, graph operations
├── retrieval/           # Retrieval strategies (vector, BM25, graph, hybrid)
├── reranking/           # Result fusion and reranking
├── memory/              # Memory layers (cache, semantic, structured)
├── query/               # Query understanding and preprocessing
├── context/             # Context building and compression
├── llm/                 # LLM clients (OpenAI, Gemini)
├── pipeline/            # Ingestion and RAG orchestration
├── api/                 # FastAPI application
├── integrations/        # External integrations (Slack)
├── utils/               # Utility functions
├── scripts/             # Setup and migration scripts
├── tests/               # Unit and integration tests
└── data/                # Data storage (FAISS index, BM25 index)
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose (for Neo4j and Redis)
- API keys for OpenAI and/or Gemini
- (Optional) Pinecone API key

### Installation

1. **Clone and setup environment**
```bash
cd cogitx-rag
cp .env.example .env
# Edit .env with your API keys
```

2. **Install dependencies**
```bash
# Using Poetry (recommended)
poetry install

# Or using pip
pip install -r requirements.txt
```

3. **Download spaCy model (for entity extraction)**
```bash
python -m spacy download en_core_web_sm
```

4. **Start infrastructure services**
```bash
docker-compose up -d
```

5. **Initialize databases**
```bash
# Setup Neo4j schema and indexes
python scripts/setup_neo4j.py

# (Optional) Setup Pinecone index
python scripts/setup_pinecone.py
```

### Running the API

```bash
# Development mode with auto-reload
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
python -m api.main
```

API documentation available at: `http://localhost:8000/docs`

## 📚 Usage

### 1. Document Ingestion

**Ingest a single document:**
```bash
curl -X POST "http://localhost:8000/ingest/document" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "doc1",
    "content": "Your document text here...",
    "metadata": {"source": "manual", "category": "tech"}
  }'
```

**Ingest document chunks (recommended for large documents):**
```bash
curl -X POST "http://localhost:8000/ingest/chunks" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "id": "chunk1",
      "document_id": "doc1",
      "content": "First chunk of text...",
      "chunk_index": 0,
      "metadata": {}
    },
    {
      "id": "chunk2",
      "document_id": "doc1",
      "content": "Second chunk of text...",
      "chunk_index": 1,
      "metadata": {}
    }
  ]'
```

### 2. Query the RAG System

```bash
curl -X POST "http://localhost:8000/query/" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "What is machine learning?",
    "user_id": "user123",
    "session_id": "session456",
    "top_k": 5,
    "include_graph": true,
    "include_memory": true
  }'
```

**Response:**
```json
{
  "query": "What is machine learning?",
  "answer": "Machine learning is a subset of artificial intelligence...",
  "sources": [
    {
      "id": "chunk1",
      "content": "Machine learning is...",
      "score": 0.95,
      "source": "vector",
      "metadata": {...}
    }
  ],
  "confidence": 0.89,
  "processing_time_ms": 245.6,
  "retrieval_stats": {
    "total_retrieved": 12,
    "top_score": 0.95
  }
}
```

## ⚙️ Configuration

Edit `.env` to configure:

### LLM Providers
```env
# Default provider: gemini or openai
DEFAULT_LLM_PROVIDER=gemini

# OpenAI
OPENAI_API_KEY=your-key
OPENAI_MODEL=gpt-4-turbo-preview

# Gemini
GEMINI_API_KEY=your-key
GEMINI_MODEL=gemini-pro
```

### Vector Store
```env
# Active store: pinecone or faiss
VECTOR_STORE_TYPE=faiss

# Pinecone (cloud)
PINECONE_API_KEY=your-key
PINECONE_INDEX_NAME=cogitx-rag

# FAISS (local)
FAISS_INDEX_PATH=./data/faiss_index
```

### Retrieval Configuration
```env
# Top-K results
VECTOR_TOP_K=10
BM25_TOP_K=10
GRAPH_MAX_RESULTS=20

# Hybrid weights
HYBRID_VECTOR_WEIGHT=0.5
HYBRID_BM25_WEIGHT=0.3
HYBRID_GRAPH_WEIGHT=0.2
```

## 🏗️ Architecture

### Retrieval Flow

```
Query Input
    │
    ├─> Query Understanding (preprocessing, expansion)
    │
    ├─> Parallel Retrieval:
    │   ├─> Vector Search (semantic similarity)
    │   ├─> BM25 Search (keyword matching)
    │   └─> Graph Traversal (multi-hop relations)
    │
    ├─> Result Fusion (RRF)
    │
    ├─> Reranking (cross-encoder)
    │
    ├─> Context Building:
    │   ├─> Retrieved documents
    │   ├─> Graph context
    │   └─> Memory context
    │
    ├─> Context Compression
    │
    ├─> LLM Generation
    │
    └─> Response + Sources
```

### Memory Layers

1. **Short-term (Redis)**: Recent queries, session history, temporary caching
2. **Long-term (Vector DB)**: Important semantic information for retrieval
3. **Structured (Neo4j)**: User preferences, metadata, structured knowledge

### Knowledge Graph

Entities and relationships extracted via spaCy NER:
- **Entities**: PERSON, ORGANIZATION, LOCATION, CONCEPT, PRODUCT, etc.
- **Relations**: MENTIONS, RELATES_TO, WORKS_FOR, CREATED_BY, etc.
- **Multi-hop traversal**: Follow entity connections up to N hops

## 🔧 Development

### Adding a New Vector Store

1. Create implementation in `vector_stores/your_store.py`:
```python
from vector_stores.base import BaseVectorStore

class YourVectorStore(BaseVectorStore):
    async def create_index(self): ...
    async def upsert(self, ...): ...
    async def search(self, ...): ...
    # ... implement all abstract methods
```

2. Register in `vector_stores/factory.py`:
```python
elif store_type == "your_store":
    return YourVectorStore(**kwargs)
```

3. Add configuration in `.env.example` and `config/settings.py`

### Adding a New LLM Provider

1. Implement in `llm/your_provider.py`:
```python
from llm.base import BaseLLM

class YourLLM(BaseLLM):
    async def generate(self, ...): ...
    async def chat(self, ...): ...
```

2. Update `api/dependencies.py` to instantiate your provider

### Testing

```bash
# Run tests
pytest

# With coverage
pytest --cov=. --cov-report=html

# Run specific test
pytest tests/unit/test_vector_stores.py
```

## 🐳 Docker Deployment

```bash
# Build and run all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 📊 Monitoring

Health check endpoints:
- `GET /health` - Overall system health
- `GET /health/dependencies` - Component status
- `GET /health/ready` - Kubernetes readiness probe
- `GET /health/live` - Kubernetes liveness probe

Metrics (if enabled):
- Prometheus metrics on port 9090
- Custom metrics for retrieval performance, LLM latency, etc.

## 🗺️ Roadmap

- [ ] Implement query decomposition for complex multi-part queries
- [ ] Add agent-based retrieval with tool calling
- [ ] Implement graph-first retrieval strategy
- [ ] Add evaluation metrics and benchmarking
- [ ] Build Streamlit UI for demos
- [ ] Add support for more LLM providers (Anthropic Claude, local models via Ollama)
- [ ] Implement document versioning and updates
- [ ] Add user feedback loop for relevance tuning

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

Built with:
- FastAPI, Pydantic, Uvicorn
- OpenAI, Google Generative AI
- Pinecone, FAISS
- Neo4j, Redis
- spaCy, sentence-transformers
- rank-bm25

---

**CogitX-RAG** - Enterprise-grade RAG with graph intelligence 🚀

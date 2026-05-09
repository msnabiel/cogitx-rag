# Production Deployment Guide

## Prerequisites

- Docker installed
- API keys: GEMINI_API_KEY or OPENAI_API_KEY
- (Optional) Neo4j instance
- (Optional) Redis instance
- (Optional) Pinecone API key

## Quick Deploy Options

### Option 1: Docker Compose (Recommended)

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 2. Start all services
docker-compose up -d

# 3. API available at
http://localhost:8000/docs
```

### Option 2: Standalone Docker

```bash
# 1. Build image
docker build -t cogitx-rag:latest .

# 2. Run container
docker run -d \
  --name cogitx-rag \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  cogitx-rag:latest

# 3. Check logs
docker logs -f cogitx-rag
```

### Option 3: Cloud Providers

#### AWS ECS/Fargate

```bash
# 1. Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker tag cogitx-rag:latest <account>.dkr.ecr.us-east-1.amazonaws.com/cogitx-rag:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/cogitx-rag:latest

# 2. Create ECS task definition (use task-definition.json)
# 3. Deploy to ECS Fargate
```

#### Google Cloud Run

```bash
# 1. Build and push
gcloud builds submit --tag gcr.io/PROJECT_ID/cogitx-rag

# 2. Deploy
gcloud run deploy cogitx-rag \
  --image gcr.io/PROJECT_ID/cogitx-rag \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --set-env-vars GEMINI_API_KEY=xxx
```

#### Railway/Render

1. Connect GitHub repo
2. Select `Dockerfile`
3. Add environment variables
4. Deploy (automatic)

### Option 4: Kubernetes

```bash
# 1. Apply configs
kubectl apply -f k8s/

# 2. Check status
kubectl get pods -n cogitx-rag

# 3. Get service URL
kubectl get service cogitx-rag -n cogitx-rag
```

## Environment Configuration

### Minimal (FAISS + Gemini)

```env
GEMINI_API_KEY=your-key
VECTOR_STORE_TYPE=faiss
DEFAULT_LLM_PROVIDER=gemini
```

### Production (Pinecone + OpenAI + Neo4j)

```env
OPENAI_API_KEY=your-key
PINECONE_API_KEY=your-key
PINECONE_INDEX_NAME=cogitx-prod
NEO4J_URI=bolt://neo4j:7687
NEO4J_PASSWORD=secure-password
REDIS_HOST=redis
VECTOR_STORE_TYPE=pinecone
DEFAULT_LLM_PROVIDER=openai
```

## Scaling

### Horizontal Scaling

```bash
# Docker Swarm
docker service scale cogitx-rag=4

# Kubernetes
kubectl scale deployment cogitx-rag --replicas=4

# Cloud Run (auto-scales)
gcloud run services update cogitx-rag --max-instances=10
```

### Resource Limits

**Minimum:**
- CPU: 1 core
- RAM: 2GB
- Storage: 5GB

**Recommended:**
- CPU: 2-4 cores
- RAM: 4-8GB
- Storage: 20GB

## Monitoring

### Health Checks

```bash
# Liveness
curl http://localhost:8000/health/live

# Readiness
curl http://localhost:8000/health/ready

# Dependencies
curl http://localhost:8000/health/dependencies
```

### Logs

```bash
# Docker
docker logs -f cogitx-rag

# Kubernetes
kubectl logs -f deployment/cogitx-rag -n cogitx-rag

# File logs
tail -f logs/cogitx-rag.log
```

### Metrics (if enabled)

```bash
# Prometheus metrics
curl http://localhost:9090/metrics
```

## Security

### Production Checklist

- [ ] Change default Neo4j password
- [ ] Use secrets management (AWS Secrets Manager, GCP Secret Manager)
- [ ] Enable HTTPS (reverse proxy with nginx/Caddy)
- [ ] Set CORS origins in production
- [ ] Use non-root container user (already configured)
- [ ] Network isolation (VPC/subnet)
- [ ] Rate limiting (API Gateway/nginx)
- [ ] API authentication (JWT/OAuth)

### SSL/TLS with Caddy (Reverse Proxy)

```bash
# docker-compose.yml add:
services:
  caddy:
    image: caddy:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile
      - caddy_data:/data
      - caddy_config:/config
```

```caddyfile
# Caddyfile
yourdomain.com {
    reverse_proxy cogitx-rag:8000
}
```

## Troubleshooting

### Container fails to start

```bash
# Check logs
docker logs cogitx-rag

# Common issues:
# 1. Missing API keys → check .env
# 2. Port conflict → change port mapping
# 3. Out of memory → increase container limits
```

### Neo4j connection fails

```bash
# Check connectivity
docker exec -it cogitx-rag python -c "from src.storage.graph.neo4j_client import Neo4jClient; import asyncio; asyncio.run(Neo4jClient().connect())"
```

### High memory usage

```bash
# Reduce workers
API_WORKERS=2

# Limit concurrent requests
MAX_WORKERS=4
```

## Cost Optimization

### LLM API Costs

- Use `max_tokens=None` cautiously
- Cache responses in Redis (already implemented)
- Use cheaper models for testing (gpt-3.5-turbo, gemini-pro)

### Infrastructure

- **Cheapest**: Railway/Render free tier + FAISS
- **Best value**: Cloud Run (pay per request) + Pinecone free tier
- **Production**: ECS/GKE + managed services

## Support

- GitHub Issues: [repo]/issues
- Docs: http://localhost:8000/docs
- Logs: `logs/cogitx-rag.log`

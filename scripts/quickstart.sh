#!/bin/bash
# Quick start script for CogitX-RAG

set -e

echo "🚀 CogitX-RAG Quick Start"
echo "=========================="

# Check if .env exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your API keys before proceeding"
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
if command -v poetry &> /dev/null; then
    poetry install
else
    pip install -r requirements.txt
fi

# Download spaCy model
echo "📥 Downloading spaCy model..."
python -m spacy download en_core_web_sm

# Start Docker services
echo "🐳 Starting Docker services (Neo4j, Redis)..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Setup databases
echo "🔧 Setting up Neo4j..."
python scripts/setup_neo4j.py

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data logs

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the API server:"
echo "  uvicorn api.main:app --reload"
echo ""
echo "API will be available at: http://localhost:8000"
echo "API docs at: http://localhost:8000/docs"
echo ""
echo "Neo4j browser: http://localhost:7474"
echo "  Username: neo4j"
echo "  Password: cogitx-password"

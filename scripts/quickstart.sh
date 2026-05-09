#!/bin/bash
# Quick start script for CogitX-RAG

set -e

echo "🚀 CogitX-RAG Quick Start"
echo "=========================="

ENV_FILE=".env"
TMP_ENV="$(mktemp)"

touch "$ENV_FILE"

prompt_var() {
    local key="$1"
    local label="$2"
    local default_value="$3"
    local current_value
    current_value="$(grep -E "^${key}=" "$ENV_FILE" | tail -n 1 | cut -d= -f2- || true)"

    if [ -n "$current_value" ]; then
        printf "%s already set\n" "$key"
        echo "${key}=${current_value}" >> "$TMP_ENV"
        return
    fi

    read -r -p "Set ${label}? [Y/n]: " should_set
    case "${should_set:-Y}" in
        n|N|no|NO)
            echo "${key}=" >> "$TMP_ENV"
            return
            ;;
    esac

    if [ -n "$default_value" ]; then
        read -r -p "${label} [${default_value}]: " value
        value="${value:-$default_value}"
    else
        read -r -p "${label}: " value
    fi

    printf '%s=%s\n' "$key" "$value" >> "$TMP_ENV"
}

echo "🧾 Collecting missing secret env values..."
prompt_var "OPENAI_API_KEY" "OpenAI API key" ""
prompt_var "GEMINI_API_KEY" "Gemini API key" ""
prompt_var "SLACK_BOT_TOKEN" "Slack bot token" ""
prompt_var "SLACK_APP_TOKEN" "Slack app token" ""
prompt_var "SLACK_SIGNING_SECRET" "Slack signing secret" ""
prompt_var "TELEGRAM_BOT_TOKEN" "Telegram bot token" ""
prompt_var "NEO4J_PASSWORD" "Neo4j password" "cogitx-password"
prompt_var "REDIS_PASSWORD" "Redis password" ""

if [ -f "$ENV_FILE" ]; then
    while IFS= read -r line; do
        case "$line" in
            \#*|"") continue ;;
            OPENAI_API_KEY=*|GEMINI_API_KEY=*|SLACK_BOT_TOKEN=*|SLACK_APP_TOKEN=*|SLACK_SIGNING_SECRET=*|TELEGRAM_BOT_TOKEN=*|NEO4J_PASSWORD=*|REDIS_PASSWORD=*)
                key="${line%%=*}"
                if ! grep -q "^${key}=" "$TMP_ENV"; then
                    printf '%s\n' "$line" >> "$TMP_ENV"
                fi
                ;;
        esac
    done < "$ENV_FILE"
fi

sort -u "$TMP_ENV" > "$ENV_FILE"
rm -f "$TMP_ENV"
echo "✅ Saved secrets to .env"

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

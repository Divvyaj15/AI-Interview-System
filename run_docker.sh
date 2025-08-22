#!/bin/bash

# AI Interview System - Docker Runner Script

echo "🐳 AI Interview System - Docker Setup"
echo "======================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not available. Please install Docker Compose."
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating template..."
    cat > .env << EOF
# AI Interview System Environment Variables
# Fill in your actual API keys below

# LLM Configuration - Choose ONE provider
LLM_MODEL=mistral/mistral-small-latest
MISTRAL_API_KEY=your_mistral_api_key_here

# Alternative: OpenAI
# LLM_MODEL=openai/gpt-3.5-turbo
# OPENAI_API_KEY=your_openai_api_key_here

# Speech-to-Text Configuration
SPEECHMATICS_API_KEY=your_speechmatics_api_key_here
EOF
    echo "✅ Created .env template. Please edit it with your API keys."
    echo "   Then run this script again."
    exit 1
fi

# Check if API keys are set
if grep -q "your_mistral_api_key_here" .env || grep -q "your_speechmatics_api_key_here" .env; then
    echo "⚠️  Please update your .env file with actual API keys before running."
    echo "   Edit the .env file and replace the placeholder values."
    exit 1
fi

echo "✅ Environment setup complete!"

# Function to stop containers
stop_containers() {
    echo "🛑 Stopping containers..."
    docker-compose down
    echo "✅ Containers stopped."
}

# Trap to stop containers on script exit
trap stop_containers EXIT

echo "🔨 Building and starting containers..."
echo "   This may take a few minutes on first run..."

# Build and start containers
docker-compose up --build

echo ""
echo "🎉 AI Interview System is running!"
echo "🌐 Open your browser and go to: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the application"


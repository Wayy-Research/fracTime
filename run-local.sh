#!/bin/bash

# Exit script if any command fails
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== FracTime Local Development Runner ===${NC}"
echo "This script runs the application locally without Kubernetes for faster development"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed.${NC}"
    exit 1
fi

# Stop any running containers
echo -e "\n${GREEN}1. Stopping any previous FracTime containers...${NC}"
docker stop fractime-app fractime-compute 2>/dev/null || true
docker rm fractime-app fractime-compute 2>/dev/null || true

# Build containers locally
echo -e "\n${GREEN}2. Building container images...${NC}"

# Build the main app
echo "Building main app..."
docker build -t fractime:local -f Dockerfile.app .

# Build the compute server
echo "Building compute server..."
docker build -t fractime-compute:local -f Dockerfile .

# Create a Docker network if it doesn't exist
echo -e "\n${GREEN}3. Setting up Docker network...${NC}"
docker network inspect fractime-network >/dev/null 2>&1 || docker network create fractime-network

# Run the compute server
echo -e "\n${GREEN}4. Starting compute server...${NC}"
docker run -d --name fractime-compute \
  --network fractime-network \
  -p 5000:5000 \
  fractime-compute:local

# Run the main app
echo -e "\n${GREEN}5. Starting main app...${NC}"
docker run -d --name fractime-app \
  --network fractime-network \
  -p 8501:8501 \
  -e COMPUTE_API_URL=http://fractime-compute:5000 \
  fractime:local

echo -e "\n${GREEN}=== Local Development Environment Ready ===${NC}"
echo "Main app is available at: http://localhost:8501"
echo "Compute API is available at: http://localhost:5000"
echo ""
echo "To view container logs:"
echo "  docker logs fractime-app -f"
echo "  docker logs fractime-compute -f"
echo ""
echo "To stop the environment:"
echo "  docker stop fractime-app fractime-compute"
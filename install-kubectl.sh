#!/bin/bash

# Exit script if any command fails
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== kubectl Installation Script ===${NC}"

# Determine OS and architecture
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

# Map architecture
case $ARCH in
    x86_64) ARCH="amd64" ;;
    aarch64) ARCH="arm64" ;;
    armv7l) ARCH="arm" ;;
    *) echo -e "${RED}Unsupported architecture: $ARCH${NC}"; exit 1 ;;
esac

echo "Detected OS: $OS, Architecture: $ARCH"

# Get latest stable kubectl version
echo -e "\n${GREEN}1. Determining latest kubectl version...${NC}"
KUBECTL_VERSION=$(curl -L -s https://dl.k8s.io/release/stable.txt)
echo "Latest stable kubectl version: $KUBECTL_VERSION"

# Download kubectl
echo -e "\n${GREEN}2. Downloading kubectl ${KUBECTL_VERSION}...${NC}"
curl -LO "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/${OS}/${ARCH}/kubectl"

# Verify the binary
echo -e "\n${GREEN}3. Verifying kubectl binary...${NC}"
curl -LO "https://dl.k8s.io/${KUBECTL_VERSION}/bin/${OS}/${ARCH}/kubectl.sha256"
echo "$(cat kubectl.sha256)  kubectl" | sha256sum --check

# Make kubectl executable
echo -e "\n${GREEN}4. Making kubectl executable...${NC}"
chmod +x kubectl

# Install kubectl
echo -e "\n${GREEN}5. Installing kubectl to /usr/local/bin/kubectl...${NC}"
sudo mv kubectl /usr/local/bin/kubectl

# Verify installation
echo -e "\n${GREEN}6. Verifying installation...${NC}"
kubectl version --client

# Clean up
rm -f kubectl.sha256

echo -e "\n${GREEN}=== kubectl Installation Complete ===${NC}"
echo "kubectl has been successfully installed to /usr/local/bin/kubectl"
echo "You can now run the deploy-to-gke.sh script to deploy your FracTime application."
#!/bin/bash

# Exit script if any command fails
set -e

# Configuration - REPLACE THESE VALUES
PROJECT_ID="wayy-research-416723"
REGION="us-central1"
CLUSTER_NAME="fractime-cluster"
IMAGE_NAME="fractime"
DOMAIN_NAME="fractime.example.com"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== FracTime GKE Deployment Script ===${NC}"
echo "This script will deploy FracTime to Google Kubernetes Engine"

# Check if Google Cloud SDK is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}Error: Google Cloud SDK (gcloud) is not installed.${NC}"
    echo "Please install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo -e "${YELLOW}Warning: kubectl is not installed.${NC}"
    echo "Please install kubectl manually before continuing:"
    echo "1. Download kubectl:"
    echo "   curl -LO \"https://dl.k8s.io/release/stable.txt/bin/linux/amd64/kubectl\""
    echo "2. Make it executable:"
    echo "   chmod +x kubectl"
    echo "3. Move to path:"
    echo "   sudo mv kubectl /usr/local/bin/"
    echo ""
    echo "Then run this script again."
    exit 1
fi

# Check if already authenticated with Google Cloud
echo -e "\n${GREEN}1. Checking Google Cloud authentication...${NC}"
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
    echo "Not authenticated. Logging in to Google Cloud..."
    gcloud auth login
else
    echo "Already authenticated as: $(gcloud auth list --filter=status:ACTIVE --format="value(account)")"
fi

# Set the current project
echo -e "\n${GREEN}2. Setting project to $PROJECT_ID...${NC}"
gcloud config set project $PROJECT_ID

# Enable required APIs
echo -e "\n${GREEN}3. Enabling required GCP APIs...${NC}"
gcloud services enable container.googleapis.com \
    containerregistry.googleapis.com \
    cloudbuild.googleapis.com

# Create a GKE cluster if it doesn't exist
echo -e "\n${GREEN}4. Checking if cluster $CLUSTER_NAME exists...${NC}"
if ! gcloud container clusters describe $CLUSTER_NAME --region $REGION &> /dev/null; then
    echo -e "${YELLOW}Cluster not found. Creating new GKE cluster...${NC}"
    gcloud container clusters create $CLUSTER_NAME \
        --region $REGION \
        --num-nodes=1 \
        --machine-type=e2-standard-2 \
        --disk-size=25 \
        --enable-autoscaling \
        --min-nodes=1 \
        --max-nodes=3
else
    echo "Cluster already exists. Using existing cluster."
fi

# Get credentials for the cluster
echo -e "\n${GREEN}5. Getting credentials for the cluster...${NC}"
gcloud container clusters get-credentials $CLUSTER_NAME --region $REGION

# Backup original Dockerfile
cp Dockerfile Dockerfile.main

# Build and push the main app Docker image using Cloud Build
echo -e "\n${GREEN}6a. Building and pushing main app Docker image using Cloud Build...${NC}"
echo "This may take a few minutes..."
gcloud builds submit --tag gcr.io/$PROJECT_ID/$IMAGE_NAME:latest .

# Build and push the compute server Docker image using Cloud Build
echo -e "\n${GREEN}6b. Building and pushing compute server Docker image using Cloud Build...${NC}"
echo "This may take a few minutes..."
cp Dockerfile.compute Dockerfile
gcloud builds submit --tag gcr.io/$PROJECT_ID/$IMAGE_NAME-compute:latest .

# Restore original Dockerfile
mv Dockerfile.main Dockerfile

# Update deployment files with correct project ID
echo -e "\n${GREEN}8. Updating Kubernetes manifests with correct project ID...${NC}"
sed -i "s|gcr.io/YOUR_PROJECT_ID/fractime|gcr.io/$PROJECT_ID/$IMAGE_NAME|g" k8s/deployment.yaml
sed -i "s|gcr.io/YOUR_PROJECT_ID/fractime-compute|gcr.io/$PROJECT_ID/$IMAGE_NAME-compute|g" k8s/compute-deployment.yaml

# Update domain name in manifests
sed -i "s|fractime.example.com|$DOMAIN_NAME|g" k8s/service.yaml
sed -i "s|fractime.example.com|$DOMAIN_NAME|g" k8s/certificate.yaml

# Create a static IP address if it doesn't exist
echo -e "\n${GREEN}9. Creating static IP address...${NC}"
if ! gcloud compute addresses describe fractime-ip --global &> /dev/null; then
    gcloud compute addresses create fractime-ip --global
fi

# Get the IP address
IP_ADDRESS=$(gcloud compute addresses describe fractime-ip --global --format='get(address)')
echo "Static IP address: $IP_ADDRESS"
echo -e "${YELLOW}Important: Set up DNS record for $DOMAIN_NAME pointing to $IP_ADDRESS${NC}"

# Apply Kubernetes manifests
echo -e "\n${GREEN}10. Applying Kubernetes manifests...${NC}"
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/compute-deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/certificate.yaml
kubectl apply -f k8s/autoscaler.yaml

# Wait for deployment to complete
echo -e "\n${GREEN}11. Waiting for deployment to complete...${NC}"
kubectl rollout status deployment/fractime-app

echo -e "\n${GREEN}=== Deployment Complete ===${NC}"
echo "Your application will be available at: https://$DOMAIN_NAME"
echo "Note: It may take a few minutes for the Ingress and Certificate to be fully provisioned."
echo "Check status with: kubectl get ingress fractime-ingress"
echo -e "${YELLOW}Important: Make sure you've configured your DNS to point $DOMAIN_NAME to $IP_ADDRESS${NC}"

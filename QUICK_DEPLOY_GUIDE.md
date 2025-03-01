# FracTime Quick Deployment Guide

This guide provides simplified instructions for deploying FracTime to Google Kubernetes Engine (GKE) for high-performance simulations.

## Prerequisites

1. **Google Cloud Account**
   - Active account with billing enabled
   - `gcloud` CLI installed (https://cloud.google.com/sdk/docs/install)

2. **Docker**
   - Docker installed and running (https://docs.docker.com/get-docker/)

## Deployment Steps

### 1. Install kubectl

If you don't have kubectl installed or had issues with the gcloud component install:

```bash
# Run the included kubectl installation script
./install-kubectl.sh
```

### 2. Configure the Deployment

Edit the project variables in `deploy-to-gke.sh`:

```bash
# Open the file
nano deploy-to-gke.sh

# Update these variables
PROJECT_ID="your-gcp-project-id"
REGION="us-central1"
CLUSTER_NAME="fractime-cluster"
DOMAIN_NAME="your-domain.example.com"
```

### 3. Run the Deployment

Execute the deployment script:

```bash
./deploy-to-gke.sh
```

The script will:
- Authenticate with Google Cloud
- Create a GKE cluster
- Build and push Docker images
- Deploy Kubernetes resources
- Configure networking and SSL

### 4. Set Up DNS

After deployment, you'll receive an IP address. Create an A record in your domain's DNS settings pointing to this IP.

### 5. Verify Deployment

Check the status of your deployment:

```bash
kubectl get pods
kubectl get services
kubectl get ingress fractime-ingress
```

## GPU Acceleration (Optional)

For heavy simulation workloads, deploy the compute backend with GPU support:

1. Build the compute image:
   ```bash
   docker build -t gcr.io/YOUR_PROJECT_ID/fractime-compute:latest -f Dockerfile.compute .
   docker push gcr.io/YOUR_PROJECT_ID/fractime-compute:latest
   ```

2. Deploy GPU-enabled compute nodes:
   ```bash
   # Edit k8s/compute-deployment.yaml with your project ID
   kubectl apply -f k8s/compute-deployment.yaml
   ```

## Troubleshooting

### Common Issues

1. **kubectl not found**:
   - Run `./install-kubectl.sh`

2. **Docker build fails**:
   - Ensure Docker is running
   - Check Docker disk space

3. **GKE cluster creation fails**:
   - Verify billing is enabled
   - Check project quotas and limits

4. **Application not accessible**:
   - Check DNS configuration
   - Verify ingress and service status

For detailed troubleshooting steps, refer to the full `GKE_DEPLOYMENT_GUIDE.md`.
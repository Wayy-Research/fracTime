# FracTime GKE Deployment Guide

This guide provides step-by-step instructions for deploying the FracTime application to Google Kubernetes Engine (GKE). This deployment is designed to handle heavy data workloads and support multiple concurrent users.

## Prerequisites

Before you begin, ensure you have the following:

1. **Google Cloud Platform Account**
   - Active billing account
   - Permissions to create GKE clusters

2. **Required Tools**
   - [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
   - [Docker](https://docs.docker.com/get-docker/)
   - [kubectl](https://kubernetes.io/docs/tasks/tools/) (can be installed via gcloud)

3. **Domain Name**
   - A domain name that you control
   - Ability to configure DNS settings

## Architecture Overview

The deployment consists of:

- **Web Frontend**: Streamlit application serving the UI
- **Backend Workers**: Python workers for computation
- **Kubernetes Resources**:
  - Deployment: Managing application pods
  - Service: Exposing the application
  - Ingress: Handling external traffic
  - HorizontalPodAutoscaler: Automatic scaling
  - ManagedCertificate: SSL/TLS certificate

## Deployment Steps

### 1. Configure the Deployment Script

Open `deploy-to-gke.sh` and modify the following variables:

```bash
PROJECT_ID="your-gcp-project-id"  # Your GCP project ID
REGION="us-central1"              # Your preferred GCP region
CLUSTER_NAME="fractime-cluster"   # Name for your GKE cluster
IMAGE_NAME="fractime"             # Container image name
DOMAIN_NAME="fractime.example.com" # Your domain name
```

### 2. Run the Deployment Script

Execute the deployment script:

```bash
./deploy-to-gke.sh
```

This script will:
- Authenticate with Google Cloud
- Create a GKE cluster if it doesn't exist
- Build and push the Docker image
- Apply Kubernetes manifests
- Configure networking and SSL

### 3. Configure DNS

After the script runs, it will output the IP address assigned to your application. Create an A record in your domain's DNS settings:

- **Type**: A
- **Name**: @ or subdomain (depending on your setup)
- **Value**: [IP_ADDRESS] (provided by the script)
- **TTL**: 300 (or as preferred)

### 4. Verify Deployment

Check the status of your deployment:

```bash
# View the deployed pods
kubectl get pods

# Check if the service is exposed correctly
kubectl get services

# Verify the ingress is properly configured
kubectl get ingress fractime-ingress

# Check certificate status
kubectl get managedcertificate
```

It may take up to 30 minutes for the SSL certificate to be provisioned. You can check its status with:

```bash
kubectl describe managedcertificate fractime-cert
```

### 5. Access the Application

Once the deployment and DNS propagation are complete, access your application at:

```
https://your-domain-name
```

## Scaling Configuration

The deployment is configured to scale automatically based on resource usage:

- **Horizontal Pod Autoscaler**: Scales from 3 to 10 replicas based on:
  - CPU usage > 70%
  - Memory usage > 80%

## Performance Tuning

For heavy data workloads, consider these optimizations:

### 1. Adjust Resource Allocations

Edit `k8s/deployment.yaml` to modify resource requests and limits:

```yaml
resources:
  requests:
    cpu: "1"       # Increase for more CPU
    memory: "2Gi"  # Increase for more memory
  limits:
    cpu: "4"
    memory: "8Gi"
```

### 2. GPU Acceleration

To add GPU support, modify `k8s/deployment.yaml`:

```yaml
# Add to container spec
resources:
  limits:
    nvidia.com/gpu: 1
```

Then create a GKE cluster with GPU nodes:

```bash
gcloud container clusters create $CLUSTER_NAME \
    --region $REGION \
    --accelerator="type=nvidia-tesla-t4,count=1" \
    --machine-type=n1-standard-8
```

### 3. Persistent Storage

For persistent data storage, add to `k8s/deployment.yaml`:

```yaml
# Add to spec
volumes:
- name: data-volume
  persistentVolumeClaim:
    claimName: fractime-data-pvc

# Add to container spec
volumeMounts:
- mountPath: "/app/data"
  name: data-volume
```

And create a PVC configuration in `k8s/storage.yaml`:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: fractime-data-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
  storageClassName: premium-rwo
```

## Monitoring and Maintenance

### View Application Logs

```bash
# Get pod names
kubectl get pods

# View logs for a specific pod
kubectl logs [pod-name]
```

### Update the Application

To update the application:

1. Make changes to your code
2. Re-run the deployment script
3. Monitor the rollout:
   ```bash
   kubectl rollout status deployment/fractime-app
   ```

### Performance Monitoring

Enable Google Cloud Monitoring to track application performance:

```bash
gcloud container clusters update $CLUSTER_NAME \
    --monitoring-service=monitoring.googleapis.com
```

## Troubleshooting

### Application Not Accessible

1. Check ingress status:
   ```bash
   kubectl describe ingress fractime-ingress
   ```

2. Verify pod status:
   ```bash
   kubectl get pods
   kubectl describe pod [pod-name]
   ```

3. Check service:
   ```bash
   kubectl get service fractime-service
   ```

### Certificate Issues

If certificate provisioning fails:

1. Check certificate status:
   ```bash
   kubectl describe managedcertificate fractime-cert
   ```

2. Verify DNS configuration is correct
3. Ensure your domain is properly resolved to the assigned IP

## Cleaning Up

To delete all deployed resources:

```bash
# Delete Kubernetes resources
kubectl delete -f k8s/

# Delete the GKE cluster
gcloud container clusters delete $CLUSTER_NAME --region $REGION

# Delete the static IP
gcloud compute addresses delete fractime-ip --global
```

## Advanced Configuration

### 1. Backend Separation

For more advanced workloads, consider separating the backend compute from the frontend:

1. Create a separate backend deployment:
   ```yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: fractime-compute
   spec:
     replicas: 2
     selector:
       matchLabels:
         app: fractime-compute
     template:
       metadata:
         labels:
           app: fractime-compute
       spec:
         containers:
         - name: compute
           image: gcr.io/YOUR_PROJECT_ID/fractime-compute:latest
           resources:
             requests:
               cpu: "2"
               memory: "4Gi"
             limits:
               cpu: "4"
               memory: "8Gi"
               nvidia.com/gpu: 1
   ```

2. Create a service for the backend:
   ```yaml
   apiVersion: v1
   kind: Service
   metadata:
     name: fractime-compute-service
   spec:
     selector:
       app: fractime-compute
     ports:
     - port: 5000
       targetPort: 5000
     type: ClusterIP
   ```

3. Modify the frontend to connect to the backend service

### 2. Cloud SQL Integration

For database storage, integrate with Cloud SQL:

1. Create a Cloud SQL instance:
   ```bash
   gcloud sql instances create fractime-db \
       --database-version=POSTGRES_13 \
       --tier=db-custom-2-7680 \
       --region=$REGION
   ```

2. Configure the application to connect to the database

### 3. Redis Cache

Add a Redis cache for improved performance:

1. Deploy Redis:
   ```yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: redis
   spec:
     selector:
       matchLabels:
         app: redis
     template:
       metadata:
         labels:
           app: redis
       spec:
         containers:
         - name: redis
           image: redis:6.2
           ports:
           - containerPort: 6379
   ---
   apiVersion: v1
   kind: Service
   metadata:
     name: redis
   spec:
     selector:
       app: redis
     ports:
     - port: 6379
       targetPort: 6379
   ```

2. Configure the application to use Redis for caching simulation results
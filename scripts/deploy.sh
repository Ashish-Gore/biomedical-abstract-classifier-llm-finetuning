#!/bin/bash

# Deployment script for the Research Paper Analysis API
# Supports deployment to various cloud platforms

set -e

# Configuration
APP_NAME="research-paper-classifier"
DOCKER_IMAGE="research-classifier"
VERSION="1.0.0"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    log_info "Docker is available"
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."
    docker build -t ${DOCKER_IMAGE}:${VERSION} .
    docker tag ${DOCKER_IMAGE}:${VERSION} ${DOCKER_IMAGE}:latest
    log_info "Docker image built successfully"
}

# Deploy locally with Docker Compose
deploy_local() {
    log_info "Deploying locally with Docker Compose..."
    docker-compose up -d
    log_info "Service deployed locally. Access at http://localhost:8000"
}

# Deploy to AWS Lambda (using AWS SAM)
deploy_aws_lambda() {
    log_info "Deploying to AWS Lambda..."
    
    # Check if AWS CLI is installed
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI is not installed. Please install AWS CLI first."
        exit 1
    fi
    
    # Check if SAM CLI is installed
    if ! command -v sam &> /dev/null; then
        log_error "AWS SAM CLI is not installed. Please install SAM CLI first."
        exit 1
    fi
    
    # Build and deploy
    sam build
    sam deploy --guided
    
    log_info "Deployed to AWS Lambda successfully"
}

# Deploy to Google Cloud Run
deploy_gcp() {
    log_info "Deploying to Google Cloud Run..."
    
    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        log_error "Google Cloud CLI is not installed. Please install gcloud first."
        exit 1
    fi
    
    # Build and push to Google Container Registry
    PROJECT_ID=$(gcloud config get-value project)
    if [ -z "$PROJECT_ID" ]; then
        log_error "No Google Cloud project set. Run 'gcloud config set project YOUR_PROJECT_ID'"
        exit 1
    fi
    
    # Build and push image
    docker build -t gcr.io/${PROJECT_ID}/${DOCKER_IMAGE}:${VERSION} .
    docker push gcr.io/${PROJECT_ID}/${DOCKER_IMAGE}:${VERSION}
    
    # Deploy to Cloud Run
    gcloud run deploy ${APP_NAME} \
        --image gcr.io/${PROJECT_ID}/${DOCKER_IMAGE}:${VERSION} \
        --platform managed \
        --region us-central1 \
        --allow-unauthenticated \
        --memory 2Gi \
        --cpu 2 \
        --max-instances 10
    
    log_info "Deployed to Google Cloud Run successfully"
}

# Deploy to Hugging Face Spaces
deploy_huggingface() {
    log_info "Deploying to Hugging Face Spaces..."
    
    # Check if huggingface_hub is installed
    if ! python -c "import huggingface_hub" &> /dev/null; then
        log_error "huggingface_hub is not installed. Install with: pip install huggingface_hub"
        exit 1
    fi
    
    log_warn "Hugging Face Spaces deployment requires manual setup:"
    log_warn "1. Create a new Space on https://huggingface.co/new-space"
    log_warn "2. Upload the Dockerfile and requirements.txt"
    log_warn "3. Configure the Space settings for Docker deployment"
    log_warn "4. The Space will automatically build and deploy"
}

# Health check
health_check() {
    log_info "Performing health check..."
    
    # Wait for service to start
    sleep 10
    
    # Check if service is responding
    if curl -f http://localhost:8000/api/v1/health &> /dev/null; then
        log_info "Service is healthy and responding"
    else
        log_error "Service health check failed"
        exit 1
    fi
}

# Main deployment function
main() {
    case "${1:-local}" in
        "local")
            check_docker
            build_image
            deploy_local
            health_check
            ;;
        "aws")
            check_docker
            build_image
            deploy_aws_lambda
            ;;
        "gcp")
            check_docker
            deploy_gcp
            ;;
        "huggingface")
            deploy_huggingface
            ;;
        "build")
            check_docker
            build_image
            ;;
        *)
            echo "Usage: $0 {local|aws|gcp|huggingface|build}"
            echo ""
            echo "  local       - Deploy locally with Docker Compose"
            echo "  aws         - Deploy to AWS Lambda"
            echo "  gcp         - Deploy to Google Cloud Run"
            echo "  huggingface - Deploy to Hugging Face Spaces"
            echo "  build       - Build Docker image only"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"

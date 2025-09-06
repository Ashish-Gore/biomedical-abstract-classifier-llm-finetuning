#!/bin/bash

# Setup script for the Research Paper Analysis Pipeline
# Installs dependencies and prepares the environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if Python is installed
check_python() {
    log_step "Checking Python installation..."
    
    if ! command -v python3 &> /dev/null; then
        if ! command -v python &> /dev/null; then
            log_error "Python is not installed. Please install Python 3.8 or higher."
            exit 1
        else
            PYTHON_CMD="python"
        fi
    else
        PYTHON_CMD="python3"
    fi
    
    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
    log_info "Found Python $PYTHON_VERSION"
    
    # Check if version is 3.8 or higher
    if ! $PYTHON_CMD -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
        log_error "Python 3.8 or higher is required. Found: $PYTHON_VERSION"
        exit 1
    fi
    
    log_info "Python version check passed"
}

# Check if pip is installed
check_pip() {
    log_step "Checking pip installation..."
    
    if ! command -v pip3 &> /dev/null; then
        if ! command -v pip &> /dev/null; then
            log_error "pip is not installed. Please install pip."
            exit 1
        else
            PIP_CMD="pip"
        fi
    else
        PIP_CMD="pip3"
    fi
    
    log_info "Found pip: $PIP_CMD"
}

# Create virtual environment
create_venv() {
    log_step "Creating virtual environment..."
    
    if [ -d "venv" ]; then
        log_warn "Virtual environment already exists. Skipping creation."
    else
        $PYTHON_CMD -m venv venv
        log_info "Virtual environment created"
    fi
    
    # Activate virtual environment
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        log_info "Virtual environment activated"
    elif [ -f "venv/Scripts/activate" ]; then
        source venv/Scripts/activate
        log_info "Virtual environment activated"
    else
        log_error "Could not activate virtual environment"
        exit 1
    fi
}

# Install Python dependencies
install_dependencies() {
    log_step "Installing Python dependencies..."
    
    # Upgrade pip
    $PIP_CMD install --upgrade pip
    
    # Install requirements
    if [ -f "requirements.txt" ]; then
        $PIP_CMD install -r requirements.txt
        log_info "Python dependencies installed"
    else
        log_error "requirements.txt not found"
        exit 1
    fi
}

# Download spaCy model
download_spacy_model() {
    log_step "Downloading spaCy model..."
    
    $PYTHON_CMD -m spacy download en_core_web_sm
    log_info "spaCy model downloaded"
}

# Create necessary directories
create_directories() {
    log_step "Creating project directories..."
    
    mkdir -p data/raw data/processed models logs notebooks tests
    log_info "Project directories created"
}

# Check CUDA availability
check_cuda() {
    log_step "Checking CUDA availability..."
    
    if $PYTHON_CMD -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null; then
        CUDA_AVAILABLE=$($PYTHON_CMD -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
        if [ "$CUDA_AVAILABLE" = "True" ]; then
            log_info "CUDA is available. GPU acceleration will be used."
        else
            log_warn "CUDA is not available. CPU will be used (slower training)."
        fi
    else
        log_warn "Could not check CUDA availability. PyTorch may not be installed yet."
    fi
}

# Run basic tests
run_tests() {
    log_step "Running basic tests..."
    
    # Test imports
    if $PYTHON_CMD -c "
import sys
sys.path.append('src')
try:
    from data_preprocessing import AbstractPreprocessor
    from disease_extraction import DiseaseExtractor
    from model_training import CancerClassifier
    from inference import ResearchPaperAnalyzer
    print('All imports successful')
except ImportError as e:
    print(f'Import error: {e}')
    sys.exit(1)
" 2>/dev/null; then
        log_info "Basic import tests passed"
    else
        log_error "Import tests failed"
        exit 1
    fi
}

# Create sample data
create_sample_data() {
    log_step "Creating sample data for testing..."
    
    $PYTHON_CMD -c "
import sys
sys.path.append('scripts')
from train_and_evaluate import create_sample_data
create_sample_data('data/raw/sample_data.csv', 100)
print('Sample data created')
" 2>/dev/null
    
    if [ -f "data/raw/sample_data.csv" ]; then
        log_info "Sample data created successfully"
    else
        log_warn "Could not create sample data"
    fi
}

# Print setup summary
print_summary() {
    echo ""
    echo "=========================================="
    echo "  SETUP COMPLETED SUCCESSFULLY!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "1. Activate virtual environment:"
    echo "   source venv/bin/activate  # Linux/Mac"
    echo "   venv\\Scripts\\activate     # Windows"
    echo ""
    echo "2. Train models:"
    echo "   python scripts/train_and_evaluate.py --create_sample_data"
    echo ""
    echo "3. Start API server:"
    echo "   python -m src.api.main"
    echo ""
    echo "4. Access API documentation:"
    echo "   http://localhost:8000/docs"
    echo ""
    echo "5. Deploy with Docker:"
    echo "   ./scripts/deploy.sh local"
    echo ""
}

# Main setup function
main() {
    echo "=========================================="
    echo "  Research Paper Analysis Pipeline Setup"
    echo "=========================================="
    echo ""
    
    check_python
    check_pip
    create_venv
    install_dependencies
    download_spacy_model
    create_directories
    check_cuda
    run_tests
    create_sample_data
    print_summary
}

# Run main function
main "$@"

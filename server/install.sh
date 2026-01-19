#!/bin/bash
# MuseTalk Server Installation Script
# This script automates the setup process for RunPod and similar environments

set -e  # Exit on error

echo "🚀 MuseTalk Server Installation Script"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the MuseTalk directory
if [ ! -f "requirements.txt" ]; then
    print_error "Please run this script from the MuseTalk root directory"
    exit 1
fi

# Step 1: Check Python version
print_info "Checking Python version..."
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -ne 3 ] || [ "$PYTHON_MINOR" -lt 9 ] || [ "$PYTHON_MINOR" -gt 10 ]; then
    print_warning "Python $PYTHON_VERSION detected. Python 3.10 is recommended."
    print_warning "Consider creating a virtual environment with Python 3.10:"
    echo "  conda create -n musetalk python=3.10 -y && conda activate musetalk"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    print_info "Python $PYTHON_VERSION detected ✓"
fi

# Step 2: Check CUDA availability
print_info "Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    print_info "CUDA is available ✓"
else
    print_warning "nvidia-smi not found. GPU acceleration may not be available."
fi

# Step 3: Install system dependencies
print_info "Installing system dependencies..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y build-essential cmake git wget ffmpeg libsm6 libxext6 libxrender-dev libgomp1 libglib2.0-0
    print_info "System dependencies installed ✓"
else
    print_warning "apt-get not found. Please install system dependencies manually."
fi

# Step 4: Install PyTorch
print_info "Installing PyTorch..."
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# Verify PyTorch installation
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'PyTorch {torch.__version__} with CUDA {torch.version.cuda} installed')"
print_info "PyTorch installed ✓"

# Step 5: Install OpenMMLab packages
print_info "Installing OpenMMLab packages..."
pip install -U openmim
mim install mmengine
mim install "mmcv==2.0.1"
mim install "mmdet==3.1.0"
mim install "mmpose==1.1.0"

# Verify mmcv installation
python -c "import mmcv; print(f'MMCV {mmcv.__version__} installed')"
print_info "OpenMMLab packages installed ✓"

# Step 6: Install MuseTalk dependencies
print_info "Installing MuseTalk dependencies..."
pip install -r requirements.txt
print_info "MuseTalk dependencies installed ✓"

# Step 7: Install server dependencies
print_info "Installing server dependencies..."
pip install -r server/requirements.txt
print_info "Server dependencies installed ✓"

# Step 8: Download models
print_info "Downloading model weights..."
if [ ! -f "download_weights.sh" ]; then
    print_error "download_weights.sh not found"
    exit 1
fi

chmod +x download_weights.sh
./download_weights.sh

# Verify models are downloaded
if [ -d "models/musetalk" ] && [ -d "models/sd-vae-ft-mse" ] && [ -d "models/whisper" ]; then
    print_info "Model weights downloaded ✓"
else
    print_error "Model download incomplete. Please check manually."
    exit 1
fi

# Step 9: Create .env file if it doesn't exist
if [ ! -f "server/.env" ] && [ -f "server/.env.example" ]; then
    print_info "Creating server/.env from template..."
    cp server/.env.example server/.env
    print_info "Configuration file created ✓"
fi

# Step 10: Test installation
print_info "Testing installation..."
python -c "
import torch
import mmcv
import mmdet
import mmpose
from transformers import WhisperModel
print('✅ All core dependencies imported successfully!')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'MMCV: {mmcv.__version__}')
"

echo ""
echo "========================================"
print_info "✅ Installation completed successfully!"
echo "========================================"
echo ""
echo "To start the server, run:"
echo "  python -m server.main"
echo ""
echo "Or use the run script:"
echo "  ./server/run.sh"
echo ""
echo "API documentation will be available at:"
echo "  http://localhost:8000/docs"
echo ""


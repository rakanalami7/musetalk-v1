# MuseTalk Server Setup Guide

This guide provides detailed instructions for setting up the MuseTalk server on RunPod or similar GPU instances.

## 🔧 System Requirements

- **GPU**: NVIDIA GPU with CUDA support (tested on RTX 4090)
- **CUDA**: 11.8 or 12.1
- **Python**: 3.10 (recommended) or 3.9
- **RAM**: 16GB+ recommended
- **Storage**: 20GB+ for models and dependencies

## 📋 Installation Steps

### Step 1: Set Up Python Environment

We **strongly recommend** using a virtual environment with Python 3.10 to avoid compilation issues with mmcv.

#### Option A: Using Conda (Recommended for RunPod)

```bash
# Install Miniconda if not already installed
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
source $HOME/miniconda/bin/activate

# Create environment with Python 3.10
conda create -n musetalk python=3.10 -y
conda activate musetalk
```

#### Option B: Using venv

```bash
# Install Python 3.10 if needed
apt-get update && apt-get install -y python3.10 python3.10-venv python3.10-dev

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate
```

### Step 2: Install System Dependencies

```bash
# Update system packages
apt-get update

# Install build essentials and other dependencies
apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0

# Install CUDA toolkit if not present (check with nvcc --version)
# For CUDA 11.8:
# wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
# sh cuda_11.8.0_520.61.05_linux.run --toolkit --silent
```

### Step 3: Clone Repository

```bash
cd /workspace  # or your preferred directory
git clone <your-musetalk-repo-url>
cd MuseTalk
```

### Step 4: Install PyTorch

Install PyTorch **before** other dependencies to ensure mmcv compiles against the correct version.

```bash
# For CUDA 11.8
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
# pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"
```

### Step 5: Install OpenMMLab Dependencies

**Important**: Install mmcv using the pre-built wheel, NOT from source!

```bash
# Install openmim (package manager for OpenMMLab)
pip install -U openmim

# Install mmcv using pre-built wheels (much faster and avoids compilation issues)
mim install mmengine
mim install "mmcv==2.0.1"
mim install "mmdet==3.1.0"
mim install "mmpose==1.1.0"

# Verify mmcv installation
python -c "import mmcv; print(f'MMCV: {mmcv.__version__}')"
```

### Step 6: Install MuseTalk Dependencies

```bash
# Install main MuseTalk requirements
pip install -r requirements.txt

# Install server-specific requirements
pip install -r server/requirements.txt
```

### Step 7: Download Model Weights

```bash
# Make download script executable
chmod +x download_weights.sh

# Download all model weights
./download_weights.sh

# Verify models are downloaded
ls -lh models/
```

Expected directory structure:
```
models/
├── dwpose/
├── face-parse-bisent/
├── musetalk/
├── sd-vae-ft-mse/
└── whisper/
```

### Step 8: Test Installation

```bash
# Test basic imports
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
```

### Step 9: Run the Server

```bash
# Start the server
python -m server.main

# Or use the run script
chmod +x server/run.sh
./server/run.sh
```

The server should start on `http://0.0.0.0:8000`

### Step 10: Test the Server

```bash
# In another terminal, test the health endpoint
curl http://localhost:8000/health

# Expected response:
# {"status":"ok","message":"MuseTalk server is running and models are loaded."}
```

## 🐛 Troubleshooting

### Issue: mmcv compilation fails with C++ errors

**Solution**: Use pre-built wheels instead of compiling from source:
```bash
pip uninstall mmcv mmcv-full -y
mim install "mmcv==2.0.1"
```

### Issue: CUDA out of memory

**Solution**: Reduce batch size in `server/config.py`:
```python
BATCH_SIZE = 4  # or even 2
```

### Issue: Models not loading

**Solution**: Check model paths and ensure all weights are downloaded:
```bash
ls -R models/
```

### Issue: Python version incompatibility

**Solution**: Use Python 3.10 specifically:
```bash
conda create -n musetalk python=3.10 -y
conda activate musetalk
```

### Issue: Import errors for MuseTalk modules

**Solution**: Ensure you're running from the MuseTalk root directory:
```bash
cd /path/to/MuseTalk
python -m server.main
```

## 🚀 Quick Start (RunPod Template)

For RunPod, you can use this one-liner setup:

```bash
# Clone and setup
git clone <repo> && cd MuseTalk && \
conda create -n musetalk python=3.10 -y && \
conda activate musetalk && \
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118 && \
pip install -U openmim && \
mim install mmengine "mmcv==2.0.1" "mmdet==3.1.0" "mmpose==1.1.0" && \
pip install -r requirements.txt && \
pip install -r server/requirements.txt && \
./download_weights.sh && \
python -m server.main
```

## 📊 Performance Tips

1. **Use Float16**: Enabled by default for 2× speedup
2. **Adjust Batch Size**: Increase for better throughput (if GPU memory allows)
3. **Pre-load Models**: Models load once at startup, not per request
4. **Use SSD Storage**: Faster model loading

## 🔐 Production Deployment

For production, consider:

1. **Use a process manager**: `supervisord` or `systemd`
2. **Enable HTTPS**: Use nginx as reverse proxy
3. **Set up monitoring**: Prometheus + Grafana
4. **Configure logging**: Centralized logging with ELK stack
5. **Add authentication**: API keys or JWT tokens

## 📝 Environment Variables

Create a `.env` file in `server/` directory:

```bash
# GPU Configuration
GPU_ID=0
USE_FLOAT16=True

# Model Configuration
VERSION=v15

# Inference Settings
BATCH_SIZE=8
FPS=25

# Server Settings
DEBUG=False
```

## 🆘 Getting Help

If you encounter issues:

1. Check the logs: `tail -f server.log`
2. Verify GPU: `nvidia-smi`
3. Check CUDA: `nvcc --version`
4. Test PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
5. Review dependencies: `pip list | grep -E "torch|mmcv|mmdet|mmpose"`

## 📚 Additional Resources

- [MuseTalk GitHub](https://github.com/TMElyralab/MuseTalk)
- [MMCV Documentation](https://mmcv.readthedocs.io/)
- [PyTorch Installation](https://pytorch.org/get-started/locally/)
- [RunPod Documentation](https://docs.runpod.io/)


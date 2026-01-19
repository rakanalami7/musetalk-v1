# MuseTalk Server - Quick Start Guide

> ⚠️ **Important**: If you encounter compilation errors (especially with `mmcv`), please see [SETUP.md](./SETUP.md) for detailed troubleshooting with Python 3.10 virtual environment setup.

## For RunPod Deployment

### Step 0: Set Up Python Environment (Recommended)

To avoid compilation issues, use Python 3.10 with a virtual environment:

```bash
# Option A: Using Conda (Recommended)
conda create -n musetalk python=3.10 -y
conda activate musetalk

# Option B: Using venv
python3.10 -m venv venv
source venv/bin/activate
```

### Step 1: Clone Repository

```bash
git clone <your-repo-url>
cd MuseTalk
```

### Step 2: Install PyTorch First

**Critical**: Install PyTorch before other dependencies to ensure mmcv compiles correctly:

```bash
# For CUDA 11.8 (most RunPod instances)
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA is available
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Step 3: Install Dependencies

```bash
# Install MMLab packages (use pre-built wheels to avoid compilation)
pip install --no-cache-dir -U openmim
mim install mmengine
mim install "mmcv==2.0.1"
mim install "mmdet==3.1.0"
mim install "mmpose==1.1.0"

# Install MuseTalk dependencies (keep versions as-is)
pip install -r requirements.txt

# Install server dependencies (separate from MuseTalk)
pip install -r server/requirements.txt
```

### Step 4: Download Models

```bash
# Download all required models
sh ./download_weights.sh
```

This will download:
- MuseTalk v1.5 UNet (~1.7GB)
- SD-VAE (~330MB)
- Whisper-tiny (~150MB)
- DWPose (~200MB)
- SyncNet (~90MB)
- Face parsing models (~95MB)

### Step 5: Configure (Optional)

```bash
# Copy environment template
cp server/.env.example server/.env

# Edit if needed (defaults work for most cases)
nano server/.env
```

### Step 6: Run Server

```bash
# Simple run
python -m server.main

# Or use the run script
sh server/run.sh
```

The server will start on `http://0.0.0.0:8000`

### Step 6: Test Server

```bash
# Health check
curl http://localhost:8000/health

# Detailed health check
curl http://localhost:8000/api/v1/health

# Check if ready
curl http://localhost:8000/api/v1/health/ready
```

### Step 7: Access API Documentation

Open in browser:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## RunPod Specific Notes

### Port Forwarding

RunPod automatically exposes ports. Your server will be accessible at:
```
https://<your-pod-id>-8000.proxy.runpod.net
```

### GPU Configuration

The server auto-detects GPU. For NVIDIA 4090:
- Default settings work great
- Float16 enabled by default
- Batch size: 20 (optimal for 4090)

### Memory Usage

Expected VRAM usage:
- Model loading: ~2-3 GB
- Inference: ~4-6 GB
- Total: ~6-8 GB (safe for 4090's 24GB)

## Testing the Server

### 1. Check Server Status

```bash
curl http://localhost:8000/
```

Expected response:
```json
{
  "message": "MuseTalk API Server",
  "version": "1.0.0",
  "musetalk_version": "v15",
  "docs": "/docs",
  "status": "running"
}
```

### 2. Check Health

```bash
curl http://localhost:8000/api/v1/health
```

Expected response:
```json
{
  "status": "healthy",
  "device": "cuda:0",
  "cuda_available": true,
  "cuda_device_count": 1,
  "cuda_device_name": "NVIDIA GeForce RTX 4090",
  "models_loaded": true,
  "model_details": {
    "vae_loaded": true,
    "unet_loaded": true,
    "whisper_loaded": true,
    "face_parser_loaded": true,
    "weight_dtype": "torch.float16"
  }
}
```

### 3. Check Readiness

```bash
curl http://localhost:8000/api/v1/health/ready
```

Expected response:
```json
{
  "ready": true,
  "message": "Server is ready to accept requests"
}
```

## Next Steps

Now that the server is running, you can:

1. **Build Inference Endpoints**: Add normal and real-time inference
2. **Add Avatar Management**: Create, list, delete avatars
3. **File Upload**: Handle video/audio uploads
4. **Task Queue**: Add Redis for async processing
5. **WebSocket**: Real-time streaming support

## Troubleshooting

### Server Won't Start

```bash
# Check Python version (should be 3.10)
python --version

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check if port is in use
lsof -i :8000
```

### Models Not Loading

```bash
# Verify model files exist
ls -la models/musetalkV15/
ls -la models/whisper/
ls -la models/dwpose/

# Re-download if needed
sh ./download_weights.sh
```

### CUDA Out of Memory

Edit `server/.env`:
```bash
BATCH_SIZE=10  # Reduce from 20
USE_FLOAT16=True  # Ensure enabled
```

### FFmpeg Not Found

```bash
# Install ffmpeg
apt-get update && apt-get install -y ffmpeg

# Verify installation
ffmpeg -version
```

## Development Mode

For development with auto-reload:

```bash
# Edit server/.env
RELOAD=True
LOG_LEVEL=DEBUG

# Run server
python -m server.main
```

## Production Deployment

For production on RunPod:

1. **Use Gunicorn** (for multiple workers):
```bash
pip install gunicorn
gunicorn server.main:app -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

2. **Use Supervisor** (for auto-restart):
```bash
apt-get install -y supervisor
# Configure supervisor to run the server
```

3. **Enable Logging**:
```bash
# In server/.env
LOG_FILE=./logs/server.log
```

## 🐛 Common Issues & Solutions

### Issue 1: mmcv compilation fails with C++ errors

**Error**: `error: no match for 'operator*'` or similar C++ compilation errors

**Solution**:
```bash
# Use Python 3.10 and pre-built wheels
conda create -n musetalk python=3.10 -y
conda activate musetalk
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
mim install "mmcv==2.0.1"  # This uses pre-built wheels, NOT source compilation
```

### Issue 2: CUDA not available

**Error**: `torch.cuda.is_available()` returns `False`

**Solution**:
```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio -y
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
```

### Issue 3: Models not loading

**Error**: `FileNotFoundError` for model files

**Solution**:
```bash
# Re-download models
chmod +x download_weights.sh
./download_weights.sh

# Verify all models exist
ls -lh models/
```

### Issue 4: Import errors for MuseTalk modules

**Error**: `ModuleNotFoundError: No module named 'musetalk'`

**Solution**:
```bash
# Ensure you're in the MuseTalk root directory
cd /path/to/MuseTalk
python -m server.main  # Use module syntax
```

### Issue 5: Python version incompatibility

**Error**: Various errors with Python 3.11+

**Solution**: Use Python 3.10 specifically (see Step 0 above)

## 📚 Support

- **Detailed Setup**: See [SETUP.md](./SETUP.md) for comprehensive troubleshooting
- **Server Documentation**: See `server/README.md`
- **Main MuseTalk docs**: See root `README.md`
- **API docs**: http://localhost:8000/docs

## Summary

✅ Server is now running and ready for endpoint development!

The foundation is set up with:
- FastAPI server with auto-docs
- Model management and initialization
- Health check endpoints
- GPU acceleration
- Configuration system
- Error handling
- Logging

Next: Build inference endpoints one by one! 🚀


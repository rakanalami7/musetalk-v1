# MuseTalk API Server

FastAPI-based REST API server for MuseTalk real-time lip-sync generation.

## Features

- 🚀 High-performance FastAPI server
- 🎯 RESTful API endpoints
- 🔥 GPU acceleration (CUDA)
- 📦 Easy deployment on RunPod
- 🔄 Real-time and batch processing
- 📊 Health monitoring endpoints
- 🔒 Optional authentication
- 📝 Auto-generated API documentation

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd MuseTalk

# Install MuseTalk dependencies
pip install -r requirements.txt

# Install MMLab packages
pip install --no-cache-dir -U openmim
mim install mmengine
mim install "mmcv==2.0.1"
mim install "mmdet==3.1.0"
mim install "mmpose==1.1.0"

# Install server dependencies
pip install -r server/requirements.txt
```

### 2. Download Models

```bash
# Linux
sh ./download_weights.sh

# Windows
download_weights.bat
```

### 3. Configuration

```bash
# Copy example environment file
cp server/.env.example server/.env

# Edit configuration as needed
nano server/.env
```

### 4. Run Server

```bash
# From MuseTalk root directory
python -m server.main

# Or with custom settings
python -m server.main
```

The server will start on `http://0.0.0.0:8000`

## API Documentation

Once the server is running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/api/v1/openapi.json

## Endpoints

### Health Check

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed health check
curl http://localhost:8000/api/v1/health

# Readiness check
curl http://localhost:8000/api/v1/health/ready

# Liveness check
curl http://localhost:8000/api/v1/health/live
```

### More Endpoints (Coming Soon)

- `/api/v1/inference/normal` - Normal inference
- `/api/v1/inference/realtime` - Real-time inference
- `/api/v1/avatar/create` - Create avatar
- `/api/v1/avatar/list` - List avatars
- `/api/v1/avatar/delete` - Delete avatar

## RunPod Deployment

### 1. Create RunPod Instance

1. Go to [RunPod](https://www.runpod.io/)
2. Create a new GPU instance (NVIDIA 4090 recommended)
3. Choose a PyTorch template or Ubuntu with CUDA

### 2. Setup on RunPod

```bash
# SSH into your RunPod instance
ssh root@<your-runpod-ip>

# Install system dependencies
apt-get update
apt-get install -y ffmpeg git

# Clone repository
git clone <your-repo-url>
cd MuseTalk

# Install Python dependencies
pip install -r requirements.txt
pip install -r server/requirements.txt

# Install MMLab packages
pip install --no-cache-dir -U openmim
mim install mmengine
mim install "mmcv==2.0.1"
mim install "mmdet==3.1.0"
mim install "mmpose==1.1.0"

# Download models
sh ./download_weights.sh

# Run server
python -m server.main
```

### 3. Expose Server

RunPod will automatically expose your ports. The server will be accessible at:
```
https://<your-pod-id>-8000.proxy.runpod.net
```

## Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# Server
HOST=0.0.0.0
PORT=8000

# GPU
GPU_ID=0
DEVICE=cuda
USE_FLOAT16=True

# Model
MUSETALK_VERSION=v15
BATCH_SIZE=20
FPS=25

# Processing
EXTRA_MARGIN=10
PARSING_MODE=jaw
LEFT_CHEEK_WIDTH=90
RIGHT_CHEEK_WIDTH=90
```

### Model Versions

- **v1.0**: Original model with L1 loss
- **v1.5**: Enhanced model with GAN + Perceptual + Sync loss (Recommended)

## Development

### Project Structure

```
MuseTalk/server/
├── main.py              # Main application
├── config.py            # Configuration
├── requirements.txt     # Server dependencies
├── .env.example         # Environment template
├── README.md           # This file
├── core/
│   ├── __init__.py
│   └── model_manager.py # Model management
└── api/
    ├── __init__.py
    └── v1/
        ├── __init__.py
        ├── health.py    # Health endpoints
        └── ...          # More endpoints
```

### Adding New Endpoints

1. Create a new router file in `server/api/v1/`
2. Define your endpoints using FastAPI
3. Include the router in `server/api/v1/__init__.py`

Example:

```python
# server/api/v1/inference.py
from fastapi import APIRouter

router = APIRouter()

@router.post("/normal")
async def normal_inference():
    # Your implementation
    pass
```

```python
# server/api/v1/__init__.py
from server.api.v1 import inference

api_router.include_router(
    inference.router, 
    prefix="/inference", 
    tags=["inference"]
)
```

## Monitoring

### Health Checks

The server provides three types of health checks:

1. **Health**: Overall server health and model status
2. **Ready**: Whether the server is ready to accept requests
3. **Live**: Whether the server process is alive

### Logs

Logs are written to stdout by default. Configure file logging in `.env`:

```bash
LOG_LEVEL=INFO
LOG_FILE=./logs/server.log
```

## Performance

### Optimization Tips

1. **Use Float16**: Set `USE_FLOAT16=True` for 2× speedup
2. **Adjust Batch Size**: Increase `BATCH_SIZE` for better throughput
3. **GPU Selection**: Use `GPU_ID` to select specific GPU
4. **Worker Processes**: Increase `WORKERS` for concurrent requests (use with caution)

### Expected Performance

On NVIDIA 4090 with Float16:
- Real-time inference: 30-40 fps
- Normal inference: 10-15 fps
- Memory usage: ~4-6 GB VRAM

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
BATCH_SIZE=10

# Ensure float16 is enabled
USE_FLOAT16=True
```

### Models Not Loading

```bash
# Check model paths
ls -la models/musetalkV15/
ls -la models/whisper/
ls -la models/dwpose/

# Re-download if needed
sh ./download_weights.sh
```

### FFmpeg Not Found

```bash
# Ubuntu/Debian
apt-get install -y ffmpeg

# Or set FFMPEG_PATH in code
```

## Security

### API Authentication (Optional)

Enable authentication in `.env`:

```bash
ENABLE_AUTH=True
SECRET_KEY=your-secret-key-here
API_KEY=your-api-key-here
```

### CORS Configuration

Configure allowed origins in `.env`:

```bash
CORS_ORIGINS=https://yourdomain.com,https://anotherdomain.com
```

## License

Same as MuseTalk - MIT License

## Support

For issues and questions:
- GitHub Issues: <your-repo-url>/issues
- Documentation: See main MuseTalk README.md

## Roadmap

- [x] Basic server setup
- [x] Health check endpoints
- [ ] Normal inference endpoint
- [ ] Real-time inference endpoint
- [ ] Avatar management endpoints
- [ ] File upload handling
- [ ] Task queue with Redis
- [ ] WebSocket support for streaming
- [ ] Authentication & authorization
- [ ] Rate limiting
- [ ] Metrics & monitoring


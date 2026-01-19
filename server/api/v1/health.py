"""
Health Check Endpoints
"""
import logging
import torch
from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Dict, Any

logger = logging.getLogger(__name__)

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    device: str
    cuda_available: bool
    cuda_device_count: int
    cuda_device_name: str = None
    models_loaded: bool
    model_details: Dict[str, Any] = None


@router.get("/", response_model=HealthResponse)
async def health_check(request: Request):
    """
    Health check endpoint
    Returns server health status and model information
    """
    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count() if cuda_available else 0
    cuda_device_name = torch.cuda.get_device_name(0) if cuda_available and cuda_device_count > 0 else None
    
    models_loaded = hasattr(request.app.state, "model_manager") and request.app.state.model_manager.is_initialized()
    
    model_details = None
    if models_loaded:
        model_details = {
            "vae_loaded": request.app.state.model_manager.vae is not None,
            "unet_loaded": request.app.state.model_manager.unet is not None,
            "whisper_loaded": request.app.state.model_manager.whisper is not None,
            "face_parser_loaded": request.app.state.model_manager.face_parser is not None,
            "weight_dtype": str(request.app.state.model_manager.weight_dtype),
        }
    
    return HealthResponse(
        status="healthy" if models_loaded else "initializing",
        device=str(request.app.state.model_manager.device) if models_loaded else "unknown",
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        cuda_device_name=cuda_device_name,
        models_loaded=models_loaded,
        model_details=model_details
    )


@router.get("/ready")
async def readiness_check(request: Request):
    """
    Readiness check endpoint
    Returns whether the server is ready to accept requests
    """
    models_loaded = hasattr(request.app.state, "model_manager") and request.app.state.model_manager.is_initialized()
    
    if models_loaded:
        return {"ready": True, "message": "Server is ready to accept requests"}
    else:
        return {"ready": False, "message": "Server is still initializing"}


@router.get("/live")
async def liveness_check():
    """
    Liveness check endpoint
    Returns whether the server is alive
    """
    return {"alive": True}


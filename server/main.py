"""
MuseTalk FastAPI Server
Main application entry point
"""
import sys
import os
from pathlib import Path

# Add parent directory to path to import musetalk modules
MUSETALK_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(MUSETALK_ROOT))

import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import torch

from server.config import settings
from server.api.v1 import api_router
from server.core.model_manager import ModelManager

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(settings.LOG_FILE) if settings.LOG_FILE else logging.NullHandler()
    ]
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_CREDENTIALS,
    allow_methods=settings.CORS_METHODS,
    allow_headers=settings.CORS_HEADERS,
    expose_headers=["*"],  # Expose all headers
    max_age=3600,  # Cache preflight requests for 1 hour
)

# Mount static files for serving results
app.mount("/results", StaticFiles(directory=str(settings.RESULTS_DIR)), name="results")

# Include API router
app.include_router(api_router, prefix=settings.API_V1_PREFIX)


@app.on_event("startup")
async def startup_event():
    """Initialize models and resources on startup"""
    logger.info("Starting MuseTalk API Server...")
    logger.info(f"Version: {settings.VERSION}")
    logger.info(f"MuseTalk Version: {settings.MUSETALK_VERSION}")
    logger.info(f"Device: {settings.DEVICE}")
    logger.info(f"GPU ID: {settings.GPU_ID}")
    logger.info(f"Use Float16: {settings.USE_FLOAT16}")
    
    # Check CUDA availability
    if settings.DEVICE == "cuda":
        if torch.cuda.is_available():
            logger.info(f"CUDA is available. GPU: {torch.cuda.get_device_name(settings.GPU_ID)}")
            logger.info(f"CUDA Version: {torch.version.cuda}")
        else:
            logger.warning("CUDA is not available. Falling back to CPU.")
            settings.DEVICE = "cpu"
    
    # Initialize model manager
    try:
        model_manager = ModelManager()
        await model_manager.initialize()
        app.state.model_manager = model_manager
        logger.info("Models initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        raise
    
    logger.info("Server startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down MuseTalk API Server...")
    
    # Cleanup model manager
    if hasattr(app.state, "model_manager"):
        await app.state.model_manager.cleanup()
    
    logger.info("Server shutdown complete")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "MuseTalk API Server",
        "version": settings.VERSION,
        "musetalk_version": settings.MUSETALK_VERSION,
        "docs": "/docs",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": settings.DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "models_loaded": hasattr(app.state, "model_manager")
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.LOG_LEVEL == "DEBUG" else "An error occurred"
        }
    )


def main():
    """Run the server"""
    uvicorn.run(
        "server.main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL.lower()
    )


if __name__ == "__main__":
    main()


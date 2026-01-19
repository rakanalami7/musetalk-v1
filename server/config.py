"""
Server Configuration
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Server settings"""
    
    # API Settings
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "MuseTalk API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "Real-time high-quality lip-sync API"
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    RELOAD: bool = False
    
    # CORS Settings
    CORS_ORIGINS: list = [
        "*",  # Allow all origins (for development)
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://localhost:3000",
    ]
    CORS_CREDENTIALS: bool = True
    CORS_METHODS: list = ["*"]
    CORS_HEADERS: list = ["*"]
    
    # File Upload Settings
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_VIDEO_EXTENSIONS: list = [".mp4", ".avi", ".mov", ".mkv"]
    ALLOWED_AUDIO_EXTENSIONS: list = [".wav", ".mp3", ".m4a", ".flac"]
    ALLOWED_IMAGE_EXTENSIONS: list = [".jpg", ".jpeg", ".png", ".bmp"]
    
    # Storage Settings
    UPLOAD_DIR: Path = Path("./uploads")
    RESULTS_DIR: Path = Path("./results")
    TEMP_DIR: Path = Path("./temp")
    
    # MuseTalk Model Settings
    MUSETALK_VERSION: str = "v15"  # v1 or v15
    UNET_MODEL_PATH: str = "./models/musetalkV15/unet.pth"
    UNET_CONFIG: str = "./models/musetalkV15/musetalk.json"
    VAE_TYPE: str = "sd-vae"
    WHISPER_DIR: str = "./models/whisper"
    USE_FLOAT16: bool = True
    BATCH_SIZE: int = 20
    FPS: int = 25
    
    # GPU Settings
    GPU_ID: int = 0
    DEVICE: str = "cuda"  # cuda or cpu
    
    # Processing Settings
    AUDIO_PADDING_LENGTH_LEFT: int = 2
    AUDIO_PADDING_LENGTH_RIGHT: int = 2
    EXTRA_MARGIN: int = 10
    PARSING_MODE: str = "jaw"  # raw or jaw
    LEFT_CHEEK_WIDTH: int = 90
    RIGHT_CHEEK_WIDTH: int = 90
    BBOX_SHIFT: int = 0  # For v1.0 compatibility
    
    # MuseTalk Root (for finding assets)
    MUSETALK_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Security Settings (optional)
    SECRET_KEY: Optional[str] = None
    API_KEY: Optional[str] = None
    ENABLE_AUTH: bool = False
    
    # Redis Settings (for task queue, optional)
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    USE_REDIS: bool = False
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()

# Create directories if they don't exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)


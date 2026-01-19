"""
Model Manager
Handles loading and managing MuseTalk models
"""
import logging
import torch
from pathlib import Path
from transformers import WhisperModel

from musetalk.utils.utils import load_all_model
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.face_parsing import FaceParsing
from server.config import settings

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages MuseTalk models and their lifecycle"""
    
    def __init__(self):
        self.device = None
        self.vae = None
        self.unet = None
        self.pe = None
        self.whisper = None
        self.audio_processor = None
        self.face_parser = None
        self.timesteps = None
        self.weight_dtype = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize all models"""
        if self.initialized:
            logger.warning("Models already initialized")
            return
        
        logger.info("Initializing models...")
        
        try:
            # Set device
            self.device = torch.device(
                f"cuda:{settings.GPU_ID}" if settings.DEVICE == "cuda" and torch.cuda.is_available() 
                else "cpu"
            )
            logger.info(f"Using device: {self.device}")
            
            # Load VAE, UNet, and Positional Encoding
            logger.info("Loading VAE, UNet, and Positional Encoding...")
            self.vae, self.unet, self.pe = load_all_model(
                unet_model_path=settings.UNET_MODEL_PATH,
                vae_type=settings.VAE_TYPE,
                unet_config=settings.UNET_CONFIG,
                device=self.device
            )
            
            # Convert to half precision if enabled
            if settings.USE_FLOAT16:
                logger.info("Converting models to float16...")
                self.pe = self.pe.half()
                self.vae.vae = self.vae.vae.half()
                self.unet.model = self.unet.model.half()
                self.weight_dtype = torch.float16
            else:
                self.weight_dtype = torch.float32
            
            # Move models to device
            self.pe = self.pe.to(self.device)
            self.vae.vae = self.vae.vae.to(self.device)
            self.unet.model = self.unet.model.to(self.device)
            
            # Set timesteps
            self.timesteps = torch.tensor([0], device=self.device)
            
            # Initialize audio processor
            logger.info("Initializing audio processor...")
            self.audio_processor = AudioProcessor(
                feature_extractor_path=settings.WHISPER_DIR
            )
            
            # Load Whisper model
            logger.info("Loading Whisper model...")
            self.whisper = WhisperModel.from_pretrained(settings.WHISPER_DIR)
            self.whisper = self.whisper.to(device=self.device, dtype=self.weight_dtype).eval()
            self.whisper.requires_grad_(False)
            
            # Initialize face parser
            logger.info("Initializing face parser...")
            if settings.MUSETALK_VERSION == "v15":
                self.face_parser = FaceParsing(
                    left_cheek_width=settings.LEFT_CHEEK_WIDTH,
                    right_cheek_width=settings.RIGHT_CHEEK_WIDTH
                )
            else:
                self.face_parser = FaceParsing()
            
            self.initialized = True
            logger.info("All models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup models and free resources"""
        logger.info("Cleaning up models...")
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Set models to None
        self.vae = None
        self.unet = None
        self.pe = None
        self.whisper = None
        self.audio_processor = None
        self.face_parser = None
        
        self.initialized = False
        logger.info("Models cleaned up")
    
    def get_models(self):
        """Get all models"""
        if not self.initialized:
            raise RuntimeError("Models not initialized. Call initialize() first.")
        
        return {
            "vae": self.vae,
            "unet": self.unet,
            "pe": self.pe,
            "whisper": self.whisper,
            "audio_processor": self.audio_processor,
            "face_parser": self.face_parser,
            "timesteps": self.timesteps,
            "device": self.device,
            "weight_dtype": self.weight_dtype
        }
    
    def is_initialized(self):
        """Check if models are initialized"""
        return self.initialized


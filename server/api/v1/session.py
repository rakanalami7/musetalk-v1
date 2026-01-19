from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict
import uuid
import os
import torch
import threading
from pathlib import Path

router = APIRouter()

# In-memory session storage (use Redis in production)
sessions: Dict[str, dict] = {}

# Global cache for pre-prepared default avatar
default_avatar_cache: Optional[Dict] = None
default_avatar_lock = threading.Lock()

class SessionCreateRequest(BaseModel):
    avatar_video_path: Optional[str] = None  # Optional custom avatar

class SessionCreateResponse(BaseModel):
    session_id: str
    status: str
    message: str

class SessionStatusResponse(BaseModel):
    session_id: str
    status: str  # "preparing", "ready", "error"
    message: str
    avatar_prepared: bool

def get_default_avatar_path():
    """Get the path to the default avatar video."""
    default_avatar = os.path.join(
        os.path.dirname(__file__), 
        "..", "..", "..", 
        "assets", "demo", "video", "Video_Portrait_Generation_Request.mp4"
    )
    return os.path.abspath(default_avatar)

@router.post("/create", response_model=SessionCreateResponse)
async def create_session(
    request: SessionCreateRequest,
    background_tasks: BackgroundTasks
):
    """
    Create a new session and prepare the avatar for generation.
    
    This endpoint:
    1. Creates a unique session ID
    2. Uses pre-prepared default avatar if available (instant)
    3. Or starts avatar preparation in background for custom avatars
    """
    global default_avatar_cache
    
    try:
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        
        # Determine avatar path
        if not request.avatar_video_path:
            avatar_path = get_default_avatar_path()
            use_default = True
        else:
            avatar_path = request.avatar_video_path
            use_default = False
        
        # Verify avatar exists
        if not os.path.exists(avatar_path):
            raise HTTPException(
                status_code=400,
                detail=f"Avatar video not found: {avatar_path}"
            )
        
        # If using default avatar and it's pre-prepared, use cached data
        if use_default and default_avatar_cache is not None:
            with default_avatar_lock:
                sessions[session_id] = {
                    "status": "ready",
                    "avatar_path": avatar_path,
                    "avatar_prepared": True,
                    "error": None,
                    "precomputed_data": default_avatar_cache.copy()
                }
            
            return SessionCreateResponse(
                session_id=session_id,
                status="ready",
                message="Session created with pre-prepared avatar. Ready to generate!"
            )
        
        # Otherwise, prepare avatar in background
        sessions[session_id] = {
            "status": "preparing",
            "avatar_path": avatar_path,
            "avatar_prepared": False,
            "error": None,
        }
        
        # Start avatar preparation in background using threading
        thread = threading.Thread(target=prepare_avatar_sync, args=(session_id, avatar_path), daemon=True)
        thread.start()
        
        return SessionCreateResponse(
            session_id=session_id,
            status="preparing",
            message="Session created. Avatar preparation started."
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{session_id}/status", response_model=SessionStatusResponse)
async def get_session_status(session_id: str):
    """
    Get the status of a session.
    
    Returns:
    - preparing: Avatar is being prepared
    - ready: Session is ready for generation
    - error: An error occurred
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    return SessionStatusResponse(
        session_id=session_id,
        status=session["status"],
        message=session.get("error", "Session is ready") if session["status"] == "error" else "Session is ready" if session["status"] == "ready" else "Preparing avatar...",
        avatar_prepared=session["avatar_prepared"]
    )

@router.delete("/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a session and clean up resources.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # TODO: Clean up any temporary files
    del sessions[session_id]
    
    return {"message": "Session deleted successfully"}

def prepare_avatar_sync(session_id: str, avatar_path: str):
    """
    Synchronous background task to prepare the avatar for generation.
    Runs in a separate thread to avoid blocking the API response.
    
    This involves:
    1. Loading the avatar video
    2. Extracting face landmarks and bounding boxes
    3. VAE encoding of all frames
    4. Pre-computing blending masks
    5. Creating cyclic frame list for temporal smoothing
    
    This follows MuseTalk's real-time inference preparation phase.
    """
    try:
        import cv2
        import numpy as np
        import pickle
        import json
        import asyncio
        from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs
        from musetalk.utils.blending import get_image_prepare_material
        from musetalk.utils.utils import get_video_fps
        from server.config import settings
        from fastapi import FastAPI
        
        # Update status
        sessions[session_id]["status"] = "preparing"
        
        # Get model_manager from app state
        # We need to import app here to avoid circular imports
        from server.main import app
        model_manager = app.state.model_manager
        
        # Get models
        models = model_manager.get_models()
        vae = models["vae"]
        device = models["device"]
        weight_dtype = models["weight_dtype"]
        face_parser = models["face_parser"]
        
        # Create session directory
        session_dir = Path(settings.RESULTS_DIR) / settings.MUSETALK_VERSION / "avatars" / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        full_imgs_path = session_dir / "full_imgs"
        full_imgs_path.mkdir(exist_ok=True)
        
        # Step 1: Extract video frames
        print(f"[Session {session_id}] Extracting video frames...")
        # Extract frames from video
        fps = get_video_fps(avatar_path)
        cap = cv2.VideoCapture(avatar_path)
        count = 0
        while True:
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(f"{full_imgs_path}/{count:08d}.png", frame)
                count += 1
            else:
                break
        cap.release()
        
        input_img_list = sorted(list(full_imgs_path.glob("*.png")))
        input_img_list = [str(img) for img in input_img_list]
        
        # Step 2: Face detection and landmark extraction
        print(f"[Session {session_id}] Detecting faces and extracting landmarks...")
        coord_list, frame_list = get_landmark_and_bbox(
            input_img_list, 
            upperbondrange=settings.BBOX_SHIFT
        )
        
        # Step 3: VAE encoding (pre-computation)
        print(f"[Session {session_id}] Encoding frames with VAE...")
        input_latent_list = []
        for bbox, frame in zip(coord_list, frame_list):
            if bbox == (-1, -1, -1, -1):  # placeholder for no face
                continue
            x1, y1, x2, y2 = bbox
            
            # Add extra margin for v1.5
            if settings.MUSETALK_VERSION == "v15":
                y2 = min(y2 + settings.EXTRA_MARGIN, frame.shape[0])
            
            # Crop and resize face
            crop_frame = frame[y1:y2, x1:x2]
            crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            
            # VAE encode
            latents = vae.get_latents_for_unet(crop_frame)
            input_latent_list.append(latents)
        
        # Step 4: Pre-compute blending masks
        print(f"[Session {session_id}] Pre-computing blending masks...")
        mask_list = []
        mask_coords_list = []
        
        mask_dir = session_dir / "mask"
        mask_dir.mkdir(exist_ok=True)
        
        parsing_mode = settings.PARSING_MODE if settings.MUSETALK_VERSION == "v15" else "raw"
        
        for idx, (bbox, frame) in enumerate(zip(coord_list, frame_list)):
            if bbox == (-1, -1, -1, -1):
                continue
            
            mask, crop_box = get_image_prepare_material(
                frame, bbox, 
                fp=face_parser,
                mode=parsing_mode
            )
            
            mask_list.append(mask)
            mask_coords_list.append(crop_box)
            
            # Save mask
            cv2.imwrite(str(mask_dir / f"{idx:08d}.png"), mask)
        
        # Step 5: Create cycle (forward + backward) for temporal smoothing
        print(f"[Session {session_id}] Creating cyclic frame list...")
        frame_list_cycle = frame_list + frame_list[::-1]
        coord_list_cycle = coord_list + coord_list[::-1]
        input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        mask_list_cycle = mask_list + mask_list[::-1]
        mask_coords_list_cycle = mask_coords_list + mask_coords_list[::-1]
        
        # Step 6: Save all pre-computed data
        print(f"[Session {session_id}] Saving pre-computed data...")
        
        # Save latents
        latents_path = session_dir / "latents.pt"
        torch.save(input_latent_list_cycle, latents_path)
        
        # Save coordinates
        coords_path = session_dir / "coords.pkl"
        with open(coords_path, 'wb') as f:
            pickle.dump(coord_list_cycle, f)
        
        # Save mask coordinates
        mask_coords_path = session_dir / "mask_coords.pkl"
        with open(mask_coords_path, 'wb') as f:
            pickle.dump(mask_coords_list_cycle, f)
        
        # Save avatar info
        avatar_info = {
            "avatar_id": session_id,
            "video_path": avatar_path,
            "num_frames": len(frame_list),
            "num_frames_cycle": len(frame_list_cycle),
            "fps": fps,
            "bbox_shift": settings.BBOX_SHIFT,
            "version": settings.MUSETALK_VERSION,
        }
        
        avatar_info_path = session_dir / "avatar_info.json"
        with open(avatar_info_path, 'w') as f:
            json.dump(avatar_info, f, indent=2)
        
        # Prepare precomputed data dict
        precomputed_data = {
            "session_dir": str(session_dir),
            "frame_list_cycle": frame_list_cycle,
            "coord_list_cycle": coord_list_cycle,
            "input_latent_list_cycle": input_latent_list_cycle,
            "mask_list_cycle": mask_list_cycle,
            "mask_coords_list_cycle": mask_coords_list_cycle,
            "num_frames": len(frame_list),
            "fps": fps,
        }
        
        # Store in session
        sessions[session_id].update({
            "status": "ready",
            "avatar_prepared": True,
            "precomputed_data": precomputed_data,
        })
        
        print(f"[Session {session_id}] Avatar preparation complete!")
        
        # If this is the default avatar, cache it globally
        if avatar_path == get_default_avatar_path():
            global default_avatar_cache
            with default_avatar_lock:
                default_avatar_cache = precomputed_data.copy()
                print(f"[Global] Default avatar cached for future sessions!")
        
    except Exception as e:
        sessions[session_id]["status"] = "error"
        sessions[session_id]["error"] = str(e)
        print(f"Error preparing avatar for session {session_id}: {e}")
        import traceback
        traceback.print_exc()


def prepare_default_avatar_at_startup():
    """
    Prepare the default avatar when the server starts.
    This runs in a background thread so it doesn't block server startup.
    """
    global default_avatar_cache
    
    print("[Startup] Preparing default avatar in background...")
    
    try:
        avatar_path = get_default_avatar_path()
        
        if not os.path.exists(avatar_path):
            print(f"[Startup] Warning: Default avatar not found at {avatar_path}")
            return
        
        # Create a temporary session ID for preparation
        temp_session_id = "default_avatar_startup"
        sessions[temp_session_id] = {
            "status": "preparing",
            "avatar_path": avatar_path,
            "avatar_prepared": False,
            "error": None,
        }
        
        # Run the preparation
        prepare_avatar_sync(temp_session_id, avatar_path)
        
        # The cache will be set automatically in prepare_avatar_sync
        print("[Startup] Default avatar preparation complete and cached!")
        
        # Clean up temp session
        del sessions[temp_session_id]
        
    except Exception as e:
        print(f"[Startup] Error preparing default avatar: {e}")
        import traceback
        traceback.print_exc()


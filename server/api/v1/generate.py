from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import Optional
import os
import tempfile
import base64
from pathlib import Path
import torch

router = APIRouter()

class GenerateRequest(BaseModel):
    session_id: str
    text: str
    voice_id: str = "JBFqnCBsd6RMkjVDRZzb"  # Default ElevenLabs voice
    model_id: str = "eleven_multilingual_v2"  # ElevenLabs TTS model

class GenerateResponse(BaseModel):
    video_url: str
    audio_url: str
    duration: float

@router.post("/text-to-video")
async def generate_text_to_video(request: GenerateRequest):
    """
    Generate avatar video with synchronized audio from text.
    
    Flow:
    1. Generate audio from text using ElevenLabs TTS
    2. Generate video using MuseTalk with the audio
    3. Return video and audio URLs
    """
    try:
        from server.api.v1.session import sessions
        
        # Verify session exists and is ready
        if request.session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[request.session_id]
        
        if session["status"] != "ready":
            raise HTTPException(
                status_code=400,
                detail=f"Session not ready. Status: {session['status']}"
            )
        
        # Step 1: Generate audio using ElevenLabs TTS
        audio_data = await generate_audio_elevenlabs(
            text=request.text,
            voice_id=request.voice_id,
            model_id=request.model_id
        )
        
        # Save audio temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as audio_file:
            audio_file.write(audio_data)
            audio_path = audio_file.name
        
        # Step 2: Generate video using MuseTalk
        video_path = await generate_video_musetalk(
            session_id=request.session_id,
            audio_path=audio_path,
            avatar_path=session["avatar_path"]
        )
        
        # Step 3: Return URLs (in production, upload to S3/CDN)
        # For now, we'll return base64 encoded data or file paths
        
        return GenerateResponse(
            video_url=f"/api/v1/generate/video/{os.path.basename(video_path)}",
            audio_url=f"/api/v1/generate/audio/{os.path.basename(audio_path)}",
            duration=0.0  # TODO: Calculate actual duration
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/text-to-video-stream")
async def generate_text_to_video_stream(request: GenerateRequest):
    """
    Stream avatar video generation in real-time.
    
    This uses ElevenLabs WebSocket TTS and MuseTalk real-time inference
    to stream video frames as they're generated.
    """
    try:
        from server.api.v1.session import sessions
        
        # Verify session
        if request.session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[request.session_id]
        
        if session["status"] != "ready":
            raise HTTPException(
                status_code=400,
                detail=f"Session not ready. Status: {session['status']}"
            )
        
        # Stream video generation
        async def video_stream_generator():
            """
            Generator that yields video chunks as they're created.
            """
            try:
                # TODO: Implement streaming generation
                # This will use:
                # 1. ElevenLabs WebSocket TTS for audio streaming
                # 2. MuseTalk real-time inference for video generation
                
                # For now, return a placeholder
                yield b"video_chunk_placeholder"
                
            except Exception as e:
                print(f"Error in video stream: {e}")
                raise
        
        return StreamingResponse(
            video_stream_generator(),
            media_type="video/mp4"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def generate_audio_elevenlabs(
    text: str,
    voice_id: str,
    model_id: str
) -> bytes:
    """
    Generate audio using ElevenLabs TTS API.
    
    Uses the streaming API for faster response.
    Reference: https://elevenlabs.io/docs/api-reference/text-to-speech
    """
    try:
        import httpx
        
        # Get API key from environment
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY not set")
        
        # ElevenLabs TTS endpoint
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": api_key
        }
        
        data = {
            "text": text,
            "model_id": model_id,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.0,
                "use_speaker_boost": True
            }
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=data, headers=headers, timeout=30.0)
            
            if response.status_code != 200:
                raise Exception(f"ElevenLabs API error: {response.status_code} - {response.text}")
            
            return response.content
    
    except Exception as e:
        print(f"Error generating audio with ElevenLabs: {e}")
        raise

async def generate_video_musetalk(
    session_id: str,
    audio_path: str,
    avatar_path: str
) -> str:
    """
    Generate video using MuseTalk with the provided audio.
    
    This uses MuseTalk's real-time inference pipeline:
    1. Extract audio features with Whisper
    2. Batch inference with UNet + VAE
    3. Blend generated faces with original frames
    4. Combine frames into video with audio
    """
    try:
        from server.api.v1.session import sessions
        import cv2
        import numpy as np
        import torch
        import math
        import subprocess
        from tqdm import tqdm
        from musetalk.utils.blending import get_image_blending
        from musetalk.utils.utils import datagen
        from server.config import settings
        
        # Get model_manager from app state
        from server.main import app
        model_manager = app.state.model_manager
        
        # Get session data
        if session_id not in sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = sessions[session_id]
        
        if session["status"] != "ready":
            raise ValueError(f"Session {session_id} not ready")
        
        # Get models
        models = model_manager.get_models()
        vae = models["vae"]
        unet = models["unet"]
        pe = models["pe"]
        whisper = models["whisper"]
        audio_processor = models["audio_processor"]
        timesteps = models["timesteps"]
        device = models["device"]
        weight_dtype = models["weight_dtype"]
        
        # Get pre-computed session data
        frame_list_cycle = session["frame_list_cycle"]
        coord_list_cycle = session["coord_list_cycle"]
        input_latent_list_cycle = session["input_latent_list_cycle"]
        mask_list_cycle = session["mask_list_cycle"]
        mask_coords_list_cycle = session["mask_coords_list_cycle"]
        
        session_dir = Path(session["session_dir"])
        
        # Create output directory
        output_dir = session_dir / "vid_output"
        output_dir.mkdir(exist_ok=True)
        
        tmp_dir = session_dir / "tmp"
        tmp_dir.mkdir(exist_ok=True)
        
        # Step 1: Extract audio features
        print(f"[Session {session_id}] Extracting audio features...")
        whisper_input_features, librosa_length = audio_processor.get_audio_feature(
            audio_path, weight_dtype=weight_dtype
        )
        
        # Step 2: Get frame-aligned audio chunks
        print(f"[Session {session_id}] Aligning audio with video frames...")
        whisper_chunks = audio_processor.get_whisper_chunk(
            whisper_input_features,
            device, weight_dtype, whisper,
            librosa_length, fps=settings.FPS,
            audio_padding_length_left=2,
            audio_padding_length_right=2
        )
        
        # Step 3: Batch inference
        print(f"[Session {session_id}] Generating lip-synced frames...")
        
        gen = datagen(
            whisper_chunks=whisper_chunks,
            vae_encode_latents=input_latent_list_cycle,
            batch_size=settings.BATCH_SIZE,
            delay_frame=0,
            device=device
        )
        
        res_frame_list = []
        
        for whisper_batch, latent_batch in tqdm(gen, desc="Generating frames"):
            # Apply positional encoding to audio
            audio_feature_batch = pe(whisper_batch.to(device))
            
            # UNet inference
            pred_latents = unet.model(
                latent_batch,
                timesteps,
                encoder_hidden_states=audio_feature_batch
            ).sample
            
            # VAE decode
            recon = vae.decode_latents(pred_latents)
            
            for res_frame in recon:
                res_frame_list.append(res_frame)
        
        # Step 4: Blend frames and save
        print(f"[Session {session_id}] Blending frames...")
        
        video_num = len(res_frame_list)
        
        for idx in tqdm(range(video_num), desc="Blending frames"):
            # Get pre-computed data (cyclic indexing)
            bbox = coord_list_cycle[idx % len(coord_list_cycle)]
            ori_frame = frame_list_cycle[idx % len(frame_list_cycle)].copy()
            mask = mask_list_cycle[idx % len(mask_list_cycle)]
            mask_crop_box = mask_coords_list_cycle[idx % len(mask_coords_list_cycle)]
            
            # Get generated frame
            res_frame = res_frame_list[idx]
            
            # Resize to bbox size
            x1, y1, x2, y2 = bbox
            
            if settings.MUSETALK_VERSION == "v15":
                y2 = min(y2 + settings.EXTRA_MARGIN, ori_frame.shape[0])
            
            res_frame = cv2.resize(
                res_frame.astype(np.uint8), 
                (x2 - x1, y2 - y1),
                interpolation=cv2.INTER_LANCZOS4
            )
            
            # Blend with original frame
            combine_frame = get_image_blending(
                ori_frame, res_frame, bbox, mask, mask_crop_box
            )
            
            # Save frame
            cv2.imwrite(str(tmp_dir / f"{idx:08d}.png"), combine_frame)
        
        # Step 5: Combine frames into video
        print(f"[Session {session_id}] Combining frames into video...")
        
        import time
        timestamp = int(time.time())
        output_video_path = output_dir / f"output_{timestamp}.mp4"
        temp_video_path = output_dir / f"temp_{timestamp}.mp4"
        
        # Create video from frames
        cmd_img2video = [
            "ffmpeg", "-y", "-v", "warning",
            "-r", str(settings.FPS),
            "-f", "image2",
            "-i", str(tmp_dir / "%08d.png"),
            "-vcodec", "libx264",
            "-vf", "format=yuv420p",
            "-crf", "18",
            str(temp_video_path)
        ]
        
        subprocess.run(cmd_img2video, check=True)
        
        # Step 6: Add audio track
        print(f"[Session {session_id}] Adding audio track...")
        
        cmd_combine_audio = [
            "ffmpeg", "-y", "-v", "warning",
            "-i", audio_path,
            "-i", str(temp_video_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-strict", "experimental",
            str(output_video_path)
        ]
        
        subprocess.run(cmd_combine_audio, check=True)
        
        # Cleanup
        temp_video_path.unlink()
        
        # Clean up tmp frames (optional, comment out for debugging)
        # import shutil
        # shutil.rmtree(tmp_dir)
        
        print(f"[Session {session_id}] Video generation complete: {output_video_path}")
        
        return str(output_video_path)
    
    except Exception as e:
        print(f"Error generating video with MuseTalk: {e}")
        import traceback
        traceback.print_exc()
        raise

@router.get("/video/{filename}")
async def serve_video(filename: str):
    """
    Serve generated video file.
    """
    # TODO: Implement proper file serving with security checks
    video_path = os.path.join(tempfile.gettempdir(), filename)
    
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")
    
    with open(video_path, "rb") as f:
        video_data = f.read()
    
    return Response(content=video_data, media_type="video/mp4")

@router.get("/audio/{filename}")
async def serve_audio(filename: str):
    """
    Serve generated audio file.
    """
    # TODO: Implement proper file serving with security checks
    audio_path = os.path.join(tempfile.gettempdir(), filename)
    
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio not found")
    
    with open(audio_path, "rb") as f:
        audio_data = f.read()
    
    return Response(content=audio_data, media_type="audio/mpeg")


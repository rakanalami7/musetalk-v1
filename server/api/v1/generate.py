from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import os
import tempfile
import base64
from pathlib import Path

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
        from server.main import model_manager
        
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
    
    This uses MuseTalk's inference pipeline to generate
    lip-synced video from audio.
    """
    try:
        # TODO: Implement MuseTalk video generation
        # This will use the models loaded in model_manager
        
        # For now, return a placeholder path
        output_path = tempfile.mktemp(suffix=".mp4")
        
        # Placeholder: copy avatar as output (replace with actual generation)
        import shutil
        shutil.copy(avatar_path, output_path)
        
        return output_path
    
    except Exception as e:
        print(f"Error generating video with MuseTalk: {e}")
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


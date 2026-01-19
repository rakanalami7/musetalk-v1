# MuseTalk Server API Documentation

## Overview

The MuseTalk server provides a RESTful API for generating lip-synced avatar videos with AI-generated speech.

**Base URL**: `https://hpdznkoepuk348-8000.proxy.runpod.net/api/v1`

## Authentication

Currently no authentication required. In production, implement API keys or JWT tokens.

## API Flow

```
1. Create Session → 2. Wait for Ready → 3. Generate Video → 4. Get Results
```

---

## Endpoints

### 1. Health Check

#### `GET /health`

Check if the server is running and models are loaded.

**Response**:
```json
{
  "status": "healthy",
  "device": "cuda",
  "cuda_available": true,
  "models_loaded": true
}
```

---

### 2. Session Management

#### `POST /api/v1/session/create`

Create a new session and prepare the avatar.

**Request Body**:
```json
{
  "avatar_video_path": null  // Optional, uses default if not provided
}
```

**Response**:
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "preparing",
  "message": "Session created. Avatar preparation started."
}
```

**Status Codes**:
- `200`: Success
- `400`: Invalid avatar path
- `500`: Server error

---

#### `GET /api/v1/session/{session_id}/status`

Get the status of a session.

**Response**:
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "ready",  // "preparing" | "ready" | "error"
  "message": "Session is ready",
  "avatar_prepared": true
}
```

**Status Codes**:
- `200`: Success
- `404`: Session not found

---

#### `DELETE /api/v1/session/{session_id}`

Delete a session and clean up resources.

**Response**:
```json
{
  "message": "Session deleted successfully"
}
```

---

### 3. Video Generation

#### `POST /api/v1/generate/text-to-video`

Generate avatar video with synchronized audio from text.

**Request Body**:
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "text": "Hello! This is a test of the MuseTalk system.",
  "voice_id": "JBFqnCBsd6RMkjVDRZzb",  // Optional, ElevenLabs voice ID
  "model_id": "eleven_multilingual_v2"  // Optional, ElevenLabs model
}
```

**Response**:
```json
{
  "video_url": "/api/v1/generate/video/output_12345.mp4",
  "audio_url": "/api/v1/generate/audio/audio_12345.mp3",
  "duration": 5.2
}
```

**Status Codes**:
- `200`: Success
- `400`: Session not ready
- `404`: Session not found
- `500`: Generation error

---

#### `POST /api/v1/generate/text-to-video-stream`

Stream avatar video generation in real-time (WebSocket-based).

**Request Body**:
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "text": "Hello! This is a test.",
  "voice_id": "JBFqnCBsd6RMkjVDRZzb",
  "model_id": "eleven_multilingual_v2"
}
```

**Response**: Streaming video/mp4

---

#### `GET /api/v1/generate/video/{filename}`

Serve a generated video file.

**Response**: video/mp4 binary data

---

#### `GET /api/v1/generate/audio/{filename}`

Serve a generated audio file.

**Response**: audio/mpeg binary data

---

## Complete Usage Example

### Step 1: Create Session

```bash
curl -X POST https://hpdznkoepuk348-8000.proxy.runpod.net/api/v1/session/create \
  -H "Content-Type: application/json" \
  -d '{}'
```

Response:
```json
{
  "session_id": "abc123",
  "status": "preparing",
  "message": "Session created. Avatar preparation started."
}
```

### Step 2: Check Session Status

```bash
curl https://hpdznkoepuk348-8000.proxy.runpod.net/api/v1/session/abc123/status
```

Keep polling until `status` is `"ready"`.

### Step 3: Generate Video

```bash
curl -X POST https://hpdznkoepuk348-8000.proxy.runpod.net/api/v1/generate/text-to-video \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "abc123",
    "text": "Hello! Welcome to MuseTalk.",
    "voice_id": "JBFqnCBsd6RMkjVDRZzb"
  }'
```

Response:
```json
{
  "video_url": "/api/v1/generate/video/output_12345.mp4",
  "audio_url": "/api/v1/generate/audio/audio_12345.mp3",
  "duration": 3.5
}
```

### Step 4: Download Results

```bash
# Download video
curl https://hpdznkoepuk348-8000.proxy.runpod.net/api/v1/generate/video/output_12345.mp4 \
  -o result.mp4

# Download audio
curl https://hpdznkoepuk348-8000.proxy.runpod.net/api/v1/generate/audio/audio_12345.mp3 \
  -o result.mp3
```

---

## JavaScript/TypeScript Example

```typescript
// 1. Create session
const createResponse = await fetch(
  'https://hpdznkoepuk348-8000.proxy.runpod.net/api/v1/session/create',
  {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({})
  }
);
const { session_id } = await createResponse.json();

// 2. Wait for session to be ready
let status = 'preparing';
while (status === 'preparing') {
  await new Promise(resolve => setTimeout(resolve, 1000)); // Wait 1 second
  
  const statusResponse = await fetch(
    `https://hpdznkoepuk348-8000.proxy.runpod.net/api/v1/session/${session_id}/status`
  );
  const statusData = await statusResponse.json();
  status = statusData.status;
}

if (status !== 'ready') {
  throw new Error('Session preparation failed');
}

// 3. Generate video
const generateResponse = await fetch(
  'https://hpdznkoepuk348-8000.proxy.runpod.net/api/v1/generate/text-to-video',
  {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id,
      text: 'Hello from MuseTalk!',
      voice_id: 'JBFqnCBsd6RMkjVDRZzb'
    })
  }
);
const { video_url, audio_url } = await generateResponse.json();

// 4. Use the URLs
console.log('Video:', video_url);
console.log('Audio:', audio_url);
```

---

## Error Responses

All error responses follow this format:

```json
{
  "error": "Error type",
  "detail": "Detailed error message"
}
```

Common error codes:
- `400`: Bad Request - Invalid parameters
- `404`: Not Found - Resource doesn't exist
- `500`: Internal Server Error - Server-side error

---

## Rate Limiting

Currently no rate limiting. In production, implement:
- 10 requests/minute per IP for session creation
- 100 requests/minute for status checks
- 5 concurrent generations per session

---

## ElevenLabs Voice IDs

Common voice IDs (see [ElevenLabs docs](https://elevenlabs.io/docs/voices/premade-voices) for full list):
- `JBFqnCBsd6RMkjVDRZzb` - George (Male, American)
- `21m00Tcm4TlvDq8ikWAM` - Rachel (Female, American)
- `AZnzlk1XvdvUeBnXmlld` - Domi (Female, American)
- `EXAVITQu4vr4xnSDxMaL` - Bella (Female, American)
- `ErXwobaYiN019PkySvjV` - Antoni (Male, American)

---

## Next Steps

1. Install httpx on the server: `pip install httpx==0.25.2`
2. Set `ELEVENLABS_API_KEY` environment variable
3. Restart the server
4. Test the endpoints!

---

## Interactive API Documentation

Visit `https://hpdznkoepuk348-8000.proxy.runpod.net/docs` for interactive Swagger UI documentation.


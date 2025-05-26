import sys
import nltk
import os
import torch
import time
from io import BytesIO
import soundfile as sf
import librosa
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn
from typing import Optional
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from starlette.background import BackgroundTask

# Download NLTK resource for English POS tagging
try:
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
except Exception as e:
    print(f"Failed to download NLTK averaged_perceptron_tagger_eng: {str(e)}")

# Mock MeCab before any imports to avoid Japanese processing errors
class MockMeCab:
    class Tagger:
        def __init__(self, *args, **kwargs):
            pass  # Do nothing
        def parse(self, text):
            return text  # Return input text unchanged (bypass Japanese processing)

sys.modules['MeCab'] = MockMeCab

# Now import other modules
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS

app = FastAPI(title="Voice Cloning API", 
              description="API for voice cloning using OpenVoice")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
# Initialize global variables (in a real production app, you'd want better state management)
target_se = None
processing = False
recorded_audio = None

# Load models at startup
device = "cuda:0" if torch.cuda.is_available() else "cpu"

try:
    tone_color_converter = ToneColorConverter('checkpoints_v2/converter/config.json', device=device)
    tone_color_converter.load_ckpt('checkpoints_v2/converter/checkpoint.pth')
    tts_model = TTS(language="EN", device=device)
    source_se_path = 'checkpoints_v2/base_speakers/ses/en-india.pth'
    
    if not os.path.exists(source_se_path):
        raise FileNotFoundError(f"Base speaker embedding not found at {source_se_path}")
    
    source_se = torch.load(source_se_path, map_location=device)
except Exception as e:
    print(f"Error loading resources: {str(e)}")
    # In production, you might want to exit or implement retry logic
    # sys.exit(1)

class TextToSpeechRequest(BaseModel):
    text: str
    speed: float = 1.0

def process_audio_to_wav(audio_data, sample_rate=16000):
    """Process audio data to WAV format in memory."""
    try:
        if isinstance(audio_data, BytesIO):
            audio_data.seek(0)
        audio, sr = librosa.load(audio_data, sr=sample_rate, mono=True)
        if len(audio) == 0:
            raise ValueError("Audio data is empty or invalid")
        output_buffer = BytesIO()
        sf.write(output_buffer, audio, sr, format='WAV')
        output_buffer.seek(0)
        return output_buffer
    except (librosa.util.exceptions.ParameterError, sf.LibsndfileError) as e:
        raise HTTPException(status_code=400, detail=f"Audio processing failed (format or data issue): {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error processing audio: {str(e)}")

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload_voice/")
async def upload_voice(file: UploadFile = File(...)):
    global target_se, recorded_audio
    
    try:
        contents = await file.read()
        audio_buffer = BytesIO(contents)
        processed_audio = process_audio_to_wav(audio_buffer)
        recorded_audio = processed_audio
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(recorded_audio.getvalue())
            temp_file_path = temp_file.name
        
        target_se, _ = se_extractor.get_se(temp_file_path, tone_color_converter, vad=True)
        
        if target_se is None:  # Additional check
            raise ValueError("Failed to extract speaker embedding")
        
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        return {"message": "Voice sample uploaded and processed successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_cloned_voice/")
async def generate_cloned_voice(request: TextToSpeechRequest):
    """Endpoint to generate cloned voice from text."""
    global target_se, recorded_audio, processing
    
    if processing:
        raise HTTPException(status_code=429, detail="Processing another request")
    
    if not recorded_audio:
        raise HTTPException(status_code=400, detail="No voice sample uploaded")
    
    if target_se is None:
        raise HTTPException(status_code=400, detail="Speaker embedding not extracted")
    
    processing = True
    start_time = time.time()
    
    try:
        # Validate speaker
        speaker_key = "EN_INDIA"
        speaker_ids = tts_model.hps.data.spk2id
        if speaker_key not in speaker_ids:
            raise HTTPException(status_code=400, 
                              detail=f"Speaker '{speaker_key}' not found. Available: {list(speaker_ids.keys())}")
        
        speaker_id = speaker_ids[speaker_key]
        
        # Generate intermediate audio
        src_path = None
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_src:
            tts_model.tts_to_file(request.text, speaker_id, temp_src.name, speed=request.speed)
            src_path = temp_src.name
        
        # Convert tone color
        out_path = None
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_out:
            tone_color_converter.convert(
                audio_src_path=src_path,
                src_se=source_se,
                tgt_se=target_se,
                output_path=temp_out.name,
                message="@MyShell"
            )
            out_path = temp_out.name
        
        # Clean up source file
        if os.path.exists(src_path):
            os.remove(src_path)
        
        # Define cleanup function
        def cleanup():
            if os.path.exists(out_path):
                os.remove(out_path)
        
        # Return the generated audio file with background task
        return FileResponse(
            out_path,
            media_type="audio/wav",
            filename="cloned_voice.wav",
            background=BackgroundTask(cleanup)  # Wrap cleanup in BackgroundTask
        )
    
    except Exception as e:
        processing = False
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        processing = False

@app.get("/status/")
async def get_status():
    """Check the status of the service."""
    return {
        "voice_uploaded": recorded_audio is not None,
        "speaker_embedding_extracted": target_se is not None,
        "processing": processing,
        "device": device
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
import time
from typing import Literal
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from src.services.piper_speaker import PIPER_VOICE_MAPPER, PiperSpeaker
from src.services.qdrant_db import QdrantDB


load_dotenv()
app = FastAPI()

db = QdrantDB()

# create class instances to load all models into memory
speaker_models = {
    key: PiperSpeaker(*PIPER_VOICE_MAPPER[key])
    for key in PIPER_VOICE_MAPPER
}


@app.get("/status")
def status():
    return {"status": "healthy"}


class ImageSearchRequest(BaseModel):
    image_data: str  # Base64 encoded image


@app.post("/search")
def search(request: ImageSearchRequest):
    results = db.search(request.image_data)
    return results


class SynthesizeRequest(BaseModel):
    text: str
    speaker: Literal['male', 'female']
    language: Literal['en', 'es', 'ca']


@app.post("/synthesize")
def synthesize(request: SynthesizeRequest):
    start_time = time.time()

    audio_array, sample_rate = speaker_models[(request.language, request.speaker)].synthesize(
        request.text
    )

    elapsed = time.time() - start_time

    return {
        "samples": audio_array.tolist(),
        "sr": sample_rate,
        "elapsed_seconds": elapsed,
        "language": request.language,
        "speaker": request.speaker,
    }

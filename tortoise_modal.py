from email import utils
import os
import io
import tempfile
import base64
from typing import Dict, List, Optional
from fastapi.params import Depends
import modal
import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_voice
import logging
import time
from pythonjsonlogger import jsonlogger
import tortoise
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
import tortoise.voices
from fastapi import Depends, FastAPI, File, Form, UploadFile, HTTPException


class UTCFormatter(jsonlogger.JsonFormatter):
    converter = time.gmtime


# Define the Modal app
app = modal.App("tortoise-tts-api")


def get_api_key(
    api_key: str = Depends(APIKeyHeader(name="tortoise_tts_api_key")),
):
    if api_key != os.environ["TORTOISE_TTS_API_KEY"]:
        raise HTTPException(status_code=403, detail="Forbidden")
    return api_key


# Create volume for model caching
model_cache_volume = modal.Volume.from_name(
    "tortoise-model-cache", create_if_missing=True
)
os.environ["TORTOISE_MODELS_DIR"] = "/model_cache"

log_volume = modal.Volume.from_name("tts_log", create_if_missing=True)

LOG_FILE_PATH = "/logs/tts.json"

# Setup logging
logging.basicConfig(level=logging.INFO)
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
formatter = UTCFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
file_handler = logging.FileHandler(LOG_FILE_PATH, mode="a")
file_handler.setFormatter(formatter)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

# Create an image with the necessary dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        "fastapi",
        "uvicorn",
        "numpy",
        "torchaudio",
        "pydantic",
        "einops",
        "transformers",
        "inflect",
        "progressbar",
        "unidecode",
        "rotary_embedding_torch",
        "python-json-logger",
    )
    .pip_install("tortoise-tts")
)


# Input model for the API
class TTSRequest(BaseModel):
    text: str
    voice: str = "random"  # Default voice
    preset: str = "fast"  # Options: ultra_fast, fast, standard, high_quality
    num_autoregressive_samples: int = Field(50, alias="numAutoregressiveSamples")
    seed: Optional[int] = None
    temperature: float = 0.8
    length_penalty: float = Field(1.0, alias="lengthPenalty")
    repetition_penalty: float = Field(2.0, alias="repetitionPenalty")
    top_p: float = Field(0.8, alias="topP")
    max_mel_tokens: int = Field(500, alias="maxMeTokens")

    class Config:
        validate_by_name = True  # Enables


# Output model for the API
class TTSResponse(BaseModel):
    audio_base64: str
    sample_rate: int


# Create FastAPI app
fastapi_app = FastAPI(title="Tortoise TTS API")

# List of built-in voices in Tortoise TTS - Focus on the most reliable ones
RELIABLE_VOICES = [
    "william",
    "train_daws",
    "geralt",
    "tim_reynolds",
    "train_empire",
    "applejack",
    "mol",
    "train_grace",
    "weaver",
    "cond_latent_example",
    "train_lescault",
    "freeman",
    "train_dotrice",
    "train_mouse",
    "pat2",
    "jlaw",
    "tom",
    "halle",
    "angie",
    "deniro",
    "train_atkins",
    "pat",
    "train_kennard",
    "rainbow",
    "emma",
    "myself",
    "snakes",
    "train_dreams",
    "lj",
    "daniel",
]


@app.function(image=image, gpu="L4", volumes={"/model_cache": model_cache_volume})
def generate_speech(request_dict):
    """Generate speech from text using Tortoise TTS with proper voice handling."""
    logger = logging.getLogger()

    # Initialize TTS model
    logger.info(f"Received request: {request_dict}")

    # Set the model cache directory
    os.makedirs("/model_cache", exist_ok=True)

    # Convert dict to proper request format
    request = request_dict

    # Validate request
    if "text" not in request or not request["text"] or request["text"].strip() == "":
        raise ValueError("The input text cannot be empty")

    # Extract parameters with defaults
    voice = request.get("voice", "tom")  # Default to "tom" instead of "random"
    preset = request.get("preset", "fast")
    seed = request.get("seed", None)
    temperature = request.get("temperature", 0.8)
    num_autoregressive_samples = request.get("num_autoregressive_samples", 50)
    length_penalty = request.get("length_penalty", 1.0)
    repetition_penalty = request.get("repetition_penalty", 2.0)
    top_p = request.get("top_p", 0.8)
    max_mel_tokens = request.get("max_mel_tokens", 500)

    logger.info(f"Using reliable voice '{voice}' instead of random")

    # Initialize TTS model
    try:

        tts = TextToSpeech(
            kv_cache=True,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        logger.info(f"TTS model initialized on device: {tts.device}")
    except Exception as e:
        logger.error(f"Failed to initialize TTS model: {str(e)}")
        raise ValueError(f"TTS model initialization failed: {str(e)}")

    # Set seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        logger.info(f"Set random seed to {seed}")

    # Validate preset
    valid_presets = ["ultra_fast", "fast", "standard", "high_quality"]
    if preset not in valid_presets:
        logger.warning(f"Invalid preset '{preset}'. Using 'fast' instead.")
        preset = "fast"

    # Generate speech
    try:
        logger.info(
            f"Generating speech for text: '{request['text'][:50]}...' with preset: {preset}"
        )
        logger.info(f"Using voice: {voice}")

        # Load voice samples
        try:
            # First try to load the requested voice
            voice_samples, conditioning_latents = load_voice(voice)
            logger.info(f"Successfully loaded requested voice: {voice}")
        except Exception as voice_error:
            # If requested voice fails, fall back to a reliable voice
            logger.warning(f"Failed to load voice '{voice}': {str(voice_error)}")

            # Try a reliable fallback voice
            backup_voice = "tom"  # Most reliable voice
            try:
                voice_samples, conditioning_latents = load_voice(backup_voice)
                logger.info(f"Falling back to reliable voice: {backup_voice}")
            except Exception as backup_error:
                logger.error(f"Failed to load backup voice: {str(backup_error)}")
                raise ValueError(f"Failed to load voice: {str(backup_error)}")

        # IMPORTANT: When using load_voice(), let Tortoise handle the conditioning internally
        tts_args = {
            "text": request["text"],
            "voice_samples": voice_samples,
            "conditioning_latents": conditioning_latents,  # This is important!
            "preset": preset,
            "num_autoregressive_samples": num_autoregressive_samples,
            "temperature": temperature,
            "length_penalty": length_penalty,
            "repetition_penalty": repetition_penalty,
            "top_p": top_p,
            "max_mel_tokens": max_mel_tokens,
            "use_deterministic_seed": seed is not None,
        }

        # Generate speech
        logger.info("Calling tts.tts_with_preset() with proper arguments")

        gen_result = tts.tts_with_preset(**tts_args)
        logger.info(f"tensor shape after gen result: {gen_result.shape}")

        if gen_result is None or gen_result.nelement() == 0:
            logger.info(f"tensor shape if is none: {gen_result.shape}")
            raise ValueError("Generated audio is empty")
        logger.info(f"Speech generation successful, tensor shape: {gen_result.shape}")

        # Convert to WAV format
        # Convert to WAV format
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
            # Handle tensor dimensions - we need to convert to 2D for torchaudio.save
            logger.info(f"Tensor shape before processing: {gen_result.shape}")

            # Make sure we have a 2D tensor (channels, samples)
            if len(gen_result.shape) == 1:
                # If 1D, add channel dimension
                gen_result = gen_result.unsqueeze(0)
            elif len(gen_result.shape) == 3:
                # If 3D [1, 1, samples], remove extra dimensions
                gen_result = gen_result.squeeze(0)  # Becomes [1, samples]

            logger.info(f"Tensor shape after processing for save: {gen_result.shape}")

            # Now save the properly shaped tensor
            torchaudio.save(temp_file.name, gen_result, 24000)

            # Read and encode
            with open(temp_file.name, "rb") as audio_file:
                audio_bytes = audio_file.read()
                audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        return {"audio_base64": audio_base64, "sample_rate": 24000}
    except Exception as e:
        error_msg = f"Error during speech generation: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)


@app.function(
    image=image,
    cpu=1.0,
    volumes={"/model_cache": model_cache_volume},
    secrets=[modal.Secret.from_name("tortoise-tts-api-key")],
)
@fastapi_app.post("/tts")
async def tts(request: TTSRequest, api_key: str = Depends(get_api_key)) -> TTSResponse:
    """Web endpoint for TTS generation."""
    try:
        # Log the incoming request
        logger.info(
            f"Received TTS request: voice={request.voice}, preset={request.preset}"
        )

        # Call the standalone function
        response_dict = generate_speech.remote(request.model_dump())
        return TTSResponse(**response_dict)
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"TTS generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")


@app.function(
    image=image,
    cpu=1.0,
    volumes={"/model_cache": model_cache_volume, "/logs": log_volume},
)
@modal.asgi_app()
def fastapi_asgi_app():
    return fastapi_app

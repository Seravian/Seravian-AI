from uuid import UUID
from fastapi.security import APIKeyHeader
import modal
from fastapi import Depends, FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from transformers import pipeline
import tempfile
import os
import torch
import subprocess
import concurrent.futures
import logging
import time
from pythonjsonlogger import jsonlogger

LOG_FILE_PATH = "/logs/ser_and_whisper.json"  # Use Modal volume mount path

# Ensure the logs directory exists (defensive check)
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)


class UTCFormatter(jsonlogger.JsonFormatter):
    converter = time.gmtime


# Persistent model cache volume
model_cache_volume = modal.Volume.from_name(
    "model-cache-volume", create_if_missing=True
)
log_volume = modal.Volume.from_name("log-volume", create_if_missing=True)
device = "cuda:0" if torch.cuda.is_available() else "cpu"


formatter = UTCFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")

file_handler = logging.FileHandler(LOG_FILE_PATH, mode="a")
file_handler.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

# Modal image with necessary dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.1.2",
        "transformers==4.38.2",
        "ffmpeg-python==0.2.0",
        "fastapi==0.110.1",
        "uvicorn==0.29.0",
        "numpy==1.26.0",
        "python-multipart==0.0.9",  # required for file upload
        "python-json-logger",
    )
    .run_commands(
        "apt-get update && "
        "apt-get install -y ffmpeg && "
        "rm -rf /var/lib/apt/lists/*"
    )
)

app = modal.App("whisper-emotion-combined")

fastapi_app = FastAPI()

emotion_classifier = None
asr_pipeline_instance = None


def get_int_value_from_emotion_str(emotion: str) -> int:
    return {
        "angry": 0,
        "happy": 1,
        "fearful": 2,
        "neutral": 3,
        "surprised": 4,
        "disgusted": 5,
        "calm": 6,
        "sad": 7,
    }.get(emotion, -1)


def get_emotion_pipeline():
    global emotion_classifier
    if emotion_classifier is None:
        emotion_classifier = pipeline(
            "audio-classification",
            model="firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3",
            device=device,
            return_all_scores=True,
            
        )
    return emotion_classifier


def get_asr_pipeline():
    global asr_pipeline_instance
    if asr_pipeline_instance is None:
        asr_pipeline_instance = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3-turbo",
            device=device,
        )
    return asr_pipeline_instance


def convert_to_wav(input_path: str, output_path: str) -> bool:
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                input_path,
                "-ac",
                "1",
                "-ar",
                "16000",
                "-acodec",
                "pcm_s16le",
                "-loglevel",
                "error",
                output_path,
            ],
            check=True,
            capture_output=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg failed: {e.stderr.decode()}")
        return False


def get_api_key(
    api_key: str = Depends(APIKeyHeader(name="whisper_emotion_combined_api_key")),
):
    if api_key != os.environ["WHISPER_EMOTION_COMBINED_API_KEY"]:
        raise HTTPException(status_code=403, detail="Forbidden")
    return api_key


@fastapi_app.post("/process")
async def process_audio(
    file: UploadFile = File(...),
    id: str = Form(...),
    api_key: str = Depends(get_api_key),
):

    valid_extensions = {
        ".wav",
        ".mp3",
        ".flac",
        ".ogg",
        ".m4a",
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
    }
    try:
        uuid = UUID(id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid GUID format")

    logger.info(f"Received file", extra={"file_name": file.filename, "id": uuid})

    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in valid_extensions:

        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported: {', '.join(valid_extensions)}",
        )

    with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as input_file:
        logger.info(f"Created temporary file: {input_file.name}")

        input_path = input_file.name

        input_file.write(await file.read())
        logger.info(
            f"Wrote temporary file",
            extra={"file_name": file.filename, "id": uuid, "input_path": input_path},
        )
    wav_path = tempfile.mktemp(suffix=".wav")
    logger.info(
        f"Created temporary wav file",
        extra={
            "wav_path": wav_path,
            "id": uuid,
            "input_path": input_path,
            "file_name": file.filename,
        },
    )
    try:
        if not convert_to_wav(input_path, wav_path):
            logger.error(
                f"Audio conversion failed",
                extra={
                    "input_path": input_path,
                    "wav_path": wav_path,
                    "id": uuid,
                    "file_name": file.filename,
                },
            )
            raise HTTPException(status_code=400, detail="Audio conversion failed")

        time_before_get_ser_pipeline = time.time()
        emotion_pipe = get_emotion_pipeline()

        time_after_get_ser_pipeline = time.time()
        logger.info(
            f"Loaded SER pipeline",
            extra={
                "id": uuid,
                "file_name": file.filename,
                "input_path": input_path,
                "wav_path": wav_path,
                "time_to_load_pipeline_in_seconds": time_after_get_ser_pipeline
                - time_before_get_ser_pipeline,
            },
        )

        time_before_get_asr_pipeline = time.time()
        asr_pipe = get_asr_pipeline()

        time_after_get_asr_pipeline = time.time()
        logger.info(
            f"Loaded ASR pipeline",
            extra={
                "id": uuid,
                "file_name": file.filename,
                "input_path": input_path,
                "wav_path": wav_path,
                "time_to_load_pipeline_in_seconds": time_after_get_asr_pipeline
                - time_before_get_asr_pipeline,
            },
        )

        def run_emotion():
            return emotion_pipe(wav_path,generate_kwargs={"language":"english"})

        def run_asr():
            return asr_pipe(wav_path,generate_kwargs={"language":"english"})

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Run both tasks in parallel
            # and wait for them to complete
            # This will block until both tasks are done
            # and will return their results
            # in the order they were submitted
            logger.info(
                f"Running emotion and ASR pipelines in parallel",
                extra={"id": uuid, "file_name": file.filename},
            )
            time_before_run = time.time()
            future_emotion = executor.submit(run_emotion)
            future_asr = executor.submit(run_asr)
            emotion_results = future_emotion.result()
            transcription_result = future_asr.result()
            time_after_run = time.time()
            logger.info(
                f"Emotion and ASR processing completed",
                extra={
                    "id": uuid,
                    "file_name": file.filename,
                    "time_to_processing_in_seconds": time_after_run - time_before_run,
                },
            )

        formatted_emotions = [
            {
                "emotion": get_int_value_from_emotion_str(emotion["label"]),
                "score": float(emotion["score"]),
            }
            for emotion in emotion_results
        ]
        formatted_emotions.sort(key=lambda x: x["score"], reverse=True)
        dominant_emotion = (
            formatted_emotions[0]["emotion"] if formatted_emotions else None
        )
        logger.info(
            f"Returning response",
            extra={
                "id": uuid,
                "file_name": file.filename,
            },
        )
        return JSONResponse(
            content={
                "transcription": transcription_result["text"],
                "emotions": formatted_emotions,
                "dominantEmotion": dominant_emotion,
            },
            status_code=200,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        for path in [input_path, wav_path]:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    logger.info(
                        f"Deleted temporary file",
                        extra={
                            "path": path,
                            "id": uuid,
                            "file_name": file.filename,
                        },
                    )
            except Exception as e:
                print(f"Failed to delete temp file {path}: {str(e)}")


@app.function(
    secrets=[modal.Secret.from_name("whisper-emotion-combined-api-key")],
    image=image,
    gpu="any",
    volumes={"/root/.cache/huggingface": model_cache_volume, "/logs": log_volume},
    timeout=600,
)
@modal.asgi_app()
def wrapper():
    return fastapi_app

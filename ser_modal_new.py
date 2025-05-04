import modal
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from transformers import pipeline
import tempfile
import os
import torch
import subprocess
import concurrent.futures

# Persistent model cache volume
volume = modal.Volume.from_name("model-cache-volume", create_if_missing=True)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

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
        "python-multipart==0.0.9"  # required for file upload
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
                "ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", "16000",
                "-acodec", "pcm_s16le", "-loglevel", "error", output_path
            ],
            check=True,
            capture_output=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg failed: {e.stderr.decode()}")
        return False


@fastapi_app.post("/process")
async def process_audio(file: UploadFile = File(...)):
    valid_extensions = {
        ".wav", ".mp3", ".flac", ".ogg", ".m4a",
        ".mp4", ".mov", ".avi", ".mkv"
    }
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in valid_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported: {', '.join(valid_extensions)}"
        )

    with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as input_file:
        input_path = input_file.name
        input_file.write(await file.read())

    wav_path = tempfile.mktemp(suffix=".wav")

    try:
        if not convert_to_wav(input_path, wav_path):
            raise HTTPException(status_code=400, detail="Audio conversion failed")

        emotion_pipe = get_emotion_pipeline()
        asr_pipe = get_asr_pipeline()

        def run_emotion():
            return emotion_pipe(wav_path)

        def run_asr():
            return asr_pipe(wav_path)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_emotion = executor.submit(run_emotion)
            future_asr = executor.submit(run_asr)
            emotion_results = future_emotion.result()
            transcription_result = future_asr.result()

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
            except Exception as e:
                print(f"Failed to delete temp file {path}: {str(e)}")


@app.function(
    image=image,
    gpu="any",
    volumes={"/root/.cache/huggingface": volume},
    timeout=600,
)
@modal.asgi_app()
def wrapper():
    return fastapi_app

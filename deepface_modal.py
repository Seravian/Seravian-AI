import logging
from fastapi import Depends, FastAPI, Form, UploadFile, HTTPException, File
from fastapi.security import APIKeyHeader
import modal
from deepface import DeepFace
from PIL import Image as PILImage
import tempfile
import os
import traceback
import io
import time
import numpy as np
from typing import List
from pythonjsonlogger import jsonlogger
import torch
from contextlib import asynccontextmanager


class UTCFormatter(jsonlogger.JsonFormatter):
    converter = time.gmtime


# Persistent model cache volume
model_cache_volume = modal.Volume.from_name(
    "model-cache-volume", create_if_missing=True
)

log_volume = modal.Volume.from_name("deep-log-volume", create_if_missing=True)


app = modal.App(name="deepface-analysis")
LOG_FILE_PATH = "/logs/deepface.json"  # Use Modal volume mount path

# Ensure the logs directory exists (defensive check)
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)


formatter = UTCFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")

file_handler = logging.FileHandler(LOG_FILE_PATH, mode="a")
file_handler.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)


deepface_image = (
    modal.Image.debian_slim()
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "ffmpeg", "libsm6", "libxext6")
    .pip_install(
        "numpy",
        "scipy",
        "opencv-python",
        "tf-keras",
        "torch",
        "tensorflow",
        "keras",
        "fastapi",
        "uvicorn",
        "pillow",
        "gdown",
        "mtcnn",
        "retina-face",
        "deepface",
        "python-multipart==0.0.9",  # Required for file uploads in FastAPI
        "python-json-logger",
    )
)


# Initialize models at startup using lifespan
@asynccontextmanager
async def lifespan(app):
    # Set environment variable to specify model directory
    os.environ["DEEPFACE_HOME"] = "/model"

    try:
        # Attempt to pre-cache models by running a sample analysis
        print("Initializing DeepFace models...")

        # Create a small sample image to initialize models
        sample = np.zeros((100, 100, 3), dtype=np.uint8)  # Dummy black image
        DeepFace.analyze(
            img_path=sample,
            actions=["emotion", "gender", "age"],
            enforce_detection=False,
            detector_backend="retinaface"
        )
        
        os.makedirs("/model/.deepface/weights", exist_ok=True)    

        os.remove(sample)
        print("DeepFace models initialized successfully")
    except Exception as e:
        print(f"Error initializing models: {str(e)}")
        print(traceback.format_exc())

    yield  # FastAPI will now process requests
  


# Create FastAPI app with the lifespan manager
fastapi_app = FastAPI(title="deepface-analysis", lifespan=lifespan)


def is_allowed_file(filename):
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_api_key(
    api_key: str = Depends(APIKeyHeader(name="deepface_api_key")),
):
    if api_key != os.environ["DEEPFACE_API_KEY"]:
        raise HTTPException(status_code=403, detail="Forbidden")
    return api_key


def get_int_value_from_emotion_str(emotion: str) -> int:

    if emotion == "angry":
        return 0
    elif emotion == "disgust":
        return 1
    elif emotion == "fear":
        return 2
    elif emotion == "happy":
        return 3
    elif emotion == "sad":
        return 4
    elif emotion == "surprise":
        return 5
    elif emotion == "neutral":
        return 6
    if emotion is None:
        return None
    return -1


def get_int_value_from_gender_str(gender: str) -> int:

    if gender == "Man":
        return 0
    elif gender == "Woman":
        return 1
    elif gender is None:
        return None
    return -1


@fastapi_app.post("/deep-analyze")
async def analyze_images(
    files: List[UploadFile] = File(...),
    ids: List[str] = Form(...),
    api_key: str = Depends(get_api_key),
):
    try:
        if not files or len(files) == 0:
            raise HTTPException(status_code=400, detail="No files provided")

        if len(files) != len(ids):
            raise HTTPException(
                status_code=400, detail="Number of files and IDs do not match"
            )
        logger.info(
            "Received images for analysis",
            extra={
                "images_count": len(files),
                "images": [
                    {"file_name": file.filename, "id": id}
                    for file, id in zip(files, ids)
                ],
            },
        )
        # Fix: Use just /model as the base directory
        os.environ["DEEPFACE_HOME"] = "/model"

        results = []

        # Process each file
        for index, (file, id) in enumerate(
            zip(files, ids)
        ):  # Fix: Use enumerate(files:

            if not is_allowed_file(file.filename):
                logger.error(f"Invalid file type: {file.filename}")
                raise HTTPException(
                    status_code=400,
                    detail="Invalid file type. Only JPG, JPEG, PNG allowed",
                )

            try:
                logger.info(f"Processing file {index}: {file.filename}")
                contents = await file.read()

                # Add detailed logging for debugging
                logger.info(
                    f"Processing file", extra={"file_name": file.filename, "id": id}
                )

                img = PILImage.open(io.BytesIO(contents))

                if img.mode != "RGB":
                    logger.info(
                        f"Converting image to RGB",
                        extra={"file_name": file.filename, "id": id},
                    )
                    img = img.convert("RGB")

                with tempfile.NamedTemporaryFile(
                    suffix=".jpg", delete=False
                ) as temp_file:
                    logger.info(
                        f"Saving temporary file",
                        extra={"file_name": file.filename, "id": id},
                    )
                    img.save(temp_file, format="JPEG")
                    temp_path = temp_file.name

                logger.info(
                    f"Saved temporary file",
                    extra={"file_name": file.filename, "id": id},
                )

                # Check if weights directory exists before analysis
                weights_dir = "/model/.deepface/weights"
                if not os.path.exists(weights_dir):
                    logger.error(f"Weights directory not found: {weights_dir}")
                    os.makedirs(weights_dir, exist_ok=True)
                    logger.info(f"Created weights directory: {weights_dir}")

                start_time = time.time()
                # Use a more reliable detector backend
                analysis_list = DeepFace.analyze(
                    img_path=temp_path,
                    actions=["emotion", "gender", "age"],
                    enforce_detection=False,
                    detector_backend="retinaface",  # More reliable detector
                )

                process_time = time.time() - start_time
                logger.info(
                    f"Complete Processing file",
                    extra={
                        "file_name": file.filename,
                        "id": id,
                        "process_time": process_time,
                    },
                )
                analysis = analysis_list[0]
                os.remove(temp_path)

                # Convert emotion scores (which may be numpy float32) to Python float
                if "emotion" in analysis:
                    emotions_list = [
                        {
                            "emotion": get_int_value_from_emotion_str(emotion),
                            "score": float(score),
                        }
                        for emotion, score in analysis["emotion"].items()
                    ]
                emotions_list.sort(key=lambda x : x["score"],reverse=True)
                result = {
                    "id": id,
                    "filename": file.filename,
                    "gender": get_int_value_from_gender_str(
                        analysis.get("dominant_gender", None)
                    ),
                    "dominantEmotion": get_int_value_from_emotion_str(
                        analysis.get("dominant_emotion", None)
                    ),
                    "age": analysis.get("age", None),
                    "emotions": emotions_list,  # JSON-serializable emotion scores
                }

                results.append(result)

            except Exception as e:
                tb = traceback.format_exc()
                logger.error(f"Error processing {file.filename}: {str(e)}\n{tb}")
                raise HTTPException(
                    status_code=400,
                    detail={"error": str(e), "traceback": tb, "success": False},
                )

        # Create a response
        response_data = {
            "results": results,
        }

        # Ensure all values are JSON serializable
        def ensure_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: ensure_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [ensure_serializable(item) for item in obj]
            else:
                return obj

        response_data = ensure_serializable(response_data)
        return response_data

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Global error in analyze_images: {str(e)}\n{tb}")
        raise HTTPException(
            status_code=500, detail={"error": str(e), "traceback": tb, "success": False}
        )


# Expose the FastAPI app through Modal
@app.function(
    image=deepface_image,
    gpu="any",
    volumes={"/model": model_cache_volume, "/logs": log_volume},
    secrets=[modal.Secret.from_name("deepface-api-key")],  # Add secrets for API key
)
@modal.asgi_app()
def wrapper():
    return fastapi_app

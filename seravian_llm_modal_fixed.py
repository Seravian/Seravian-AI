import json
from typing import List, Dict
from pydantic import BaseModel, Field
import modal
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from fastapi import Depends, FastAPI, File, Form, UploadFile, HTTPException
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
import os


class ChatRequest(BaseModel):
    history: List[Dict[str, str]]


class ChatRequestVersion2(BaseModel):
    message: str
    chat_id: str = Field(..., alias="chatId")

    class Config:
        validate_by_name = True  # Enables


class ChatResponse(BaseModel):
    response: str


def get_api_key(
    api_key: str = Depends(APIKeyHeader(name="mentallama_api_key")),
):
    if api_key != os.environ["MENTALLAMA_API_KEY"]:
        raise HTTPException(status_code=403, detail="Forbidden")
    return api_key


# Create a persistent volume to store models
model_cache_volume = modal.Volume.from_name(
    "model-cache-volume", create_if_missing=True
)

chat_history_volume = modal.Volume.from_name(
    "chat-history-volume", create_if_missing=True
)

# Create the image with dependencies
image = (
    modal.Image.debian_slim()
    .apt_install("build-essential", "cmake", "g++", "git")  # Add build tools
    .pip_install("fastapi", "uvicorn")  # FastAPI dependencies
    .pip_install(
        "torch", "transformers", "pydantic", "accelerate", "BitsandBytes"
    )  # Install these first
    .pip_install("sentencepiece")  # Install separately after build tools
)

# Create a Modal Stub (this is your app)
app = modal.App("mentallama-chat-7b")

# Model name to use
model_name = "klyang/MentaLLaMA-chat-7B"
model_path = "/model"
chat_history_path = "/chat-history"


# Initialize tokenizer and model once per container
@app.function(
    image=image,
    gpu="A100",
    timeout=600,
    volumes={model_path: model_cache_volume},
)
def load_model():
    # Check if model is already cached in volume
    if not os.path.exists(f"{model_path}/config.json"):
        print(f"Downloading model {model_name} to volume...")
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(model_path)

        # Initialize model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            # quantization_config=quantisation_config,
            # offload_folder="offload",
        )
        model.eval()
        model.save_pretrained(model_path)
        print("Model downloaded and saved to volume successfully!")
    else:
        print("Model already cached in volume")


# Function to generate response
@app.function(
    image=image,
    gpu="A100",
    timeout=600,
    volumes={model_path: model_cache_volume},
)
def generate_response(conversation_history):
    """
    Generate a response based on the conversation history and user message.
    """

    # Load tokenizer and model from volume
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        # offload_folder="offload",
        # quantization_config=quantisation_config
    )
    model.eval()

    # Format history for the model
    chat_input = (
        "".join(f"{turn['role']}: {turn['content']}\n" for turn in conversation_history)
        + "assistant:"
    )

    # Tokenize and generate response
    inputs = tokenizer(chat_input, return_tensors="pt", truncation=True).to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,  # Randomness
        top_p=0.9,  # Nucleus sampling
        repetition_penalty=1.2,  # Penalize repetition
    )
    response: str = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract assistant's response
    assistant_response = response.split("assistant:")[-1].strip()

    # Clear memory
    del model
    del tokenizer
    torch.cuda.empty_cache()

    return assistant_response


@app.function(
    image=image,
    gpu="A100",
    timeout=600,
    volumes={model_path: model_cache_volume, chat_history_path: chat_history_volume},
)
def generate_response_version2(message: str, chat_id: str):
    """
    Generate a response based on the conversation history and user message.
    """

    # Load tokenizer and model from volume
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        # offload_folder="offload",
        # quantization_config=quantisation_config
    )
    model.eval()

    # region load history from local volume by chat_id as the filename.txt and create file if it doesn't exist
    # each line of file should be user: messageplaceholder or ai: responseplaceholder

    filename = f"{chat_history_path}/{chat_id}.txt"

    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            try:
                conversation_history: list[dict] = json.load(f)

            except json.JSONDecodeError:
                conversation_history: list[dict] = []
    else:
        conversation_history: list[dict] = []

    conversation_history.append({"role": "user", "content": message})

    # endregion

    # Format history for the model
    chat_input = (
        "".join(f"{turn['role']}: {turn['content']}\n" for turn in conversation_history)
        + "assistant:"
    )

    # Tokenize and generate response
    inputs = tokenizer(chat_input, return_tensors="pt", truncation=True).to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,  # Randomness
        top_p=0.9,  # Nucleus sampling
        repetition_penalty=1.2,  # Penalize repetition
    )
    response: str = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract assistant's response
    assistant_response = response.split("assistant:")[-1].strip()

    # Clear memory
    del model
    del tokenizer
    torch.cuda.empty_cache()

    conversation_history.append({"role": "assistant", "content": assistant_response})

    # Step 3: Write updated list back to the file

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(conversation_history, f, indent=4, ensure_ascii=False)
    # endregion

    return assistant_response


# Define the FastAPI endpoint
@app.function(
    image=image,
    volumes={model_path: model_cache_volume},
    secrets=[modal.Secret.from_name("mentallama-api-key")],
)
@modal.asgi_app()
def seravian_llm():
    fastapi_app = FastAPI(
        title="MentaLLaMA Chat API",
        description="API for interacting with MentaLLaMA LLM",
        version="1.0.0",
        api_key=Depends(get_api_key),
    )

    # Ensure model is loaded in the volume
    load_model.remote()

    @fastapi_app.post("/", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        try:
            response = generate_response.remote(request.history)
            return ChatResponse(response=response)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error generating response: {str(e)}"
            )

    @fastapi_app.post("/v2", response_model=ChatResponse)
    async def chat(request: ChatRequestVersion2):

        try:
            response = generate_response_version2.remote(
                request.message, request.chat_id
            )
            return ChatResponse(response=response)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error generating response: {str(e)}"
            )

    return fastapi_app

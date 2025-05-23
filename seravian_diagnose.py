import json
from typing import List, Dict
from pydantic import BaseModel, Field
import modal
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from fastapi import Depends, FastAPI, File, Form, UploadFile, HTTPException
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
import os
import datetime


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
diagnosis_volume = modal.Volume.from_name(
    "diagnosis-history-volume", create_if_missing=True
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
app = modal.App("mentallama-diagnose")

# Model name to use
model_name = "klyang/MentaLLaMA-chat-7B"
model_path = "/model"
chat_history_path = "/chat-history"
diagnosis_path="/diagnosis-history"


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
        model.save_pretrained(model_path)
        print("Model downloaded and saved to volume successfully!")
    else:
        print("Model already cached in volume")



@app.function(
    image=image,
    gpu="A100",
    timeout=600,
    volumes={model_path: model_cache_volume, chat_history_path: chat_history_volume,diagnosis_path:diagnosis_volume},
)
def generate_diagnosis(message: str, chat_id: str):
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

    filename = f"{chat_history_path}/{chat_id}.json"
    diagnosis_filename = f"{diagnosis_path}/{chat_id}.json"

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
    chat_input = "".join(f"{turn['role']}: {turn['content']}\n" for turn in conversation_history) + "assistant:"


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

    diagnosis_entry = {
        "chat_id": chat_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "user_message": message,
        "diagnosis_response": assistant_response,
        "conversation_context": conversation_history  # Original chat history without diagnosis interaction
    }

    # Load existing diagnoses for this chat_id if they exist
    if os.path.exists(diagnosis_filename):
        with open(diagnosis_filename, "r", encoding="utf-8") as f:
            try:
                diagnoses_history = json.load(f)
                if not isinstance(diagnoses_history, list):
                    diagnoses_history = [diagnoses_history]  # Convert old format to list
            except json.JSONDecodeError:
                diagnoses_history = []
    else:
        diagnoses_history = []

    # Add new diagnosis entry
    diagnoses_history.append(diagnosis_entry)

    # Save updated diagnoses to diagnosis volume
    with open(diagnosis_filename, "w", encoding="utf-8") as f:
        json.dump(diagnoses_history, f, indent=4, ensure_ascii=False)

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
        title="MentaLLaMA diagnosis API",
        description="API getting diagnosis through MentalLAMa",
        version="1.0.0",
        api_key=Depends(get_api_key),
    )

    # Ensure model is loaded in the volume
    load_model.remote()

    
    @fastapi_app.post("/get-diagnosis", response_model=ChatResponse)
    async def chat(request: ChatRequestVersion2):

        try:
            response = generate_diagnosis.remote(
                request.message, request.chat_id
            )
            return ChatResponse(response=response)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error generating response: {str(e)}"
            )

    return fastapi_app
import json
import threading
from typing import List, Dict
from pydantic import BaseModel, Field
import modal
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from fastapi import Depends, FastAPI, File, Form, Response, UploadFile, HTTPException
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
import os
import datetime
import time

# class ChatRequest(BaseModel):
#     history: List[Dict[str, str]]


class ChatRequestVersion2(BaseModel):
    message_id: int = Field(..., alias="messageId")
    message: str
    chat_id: str = Field(..., alias="chatId")

    class Config:
        validate_by_name = True  # Enables


class DeleteChatRequestVersion2(BaseModel):
    chat_id: str = Field(..., alias="chatId")

    class Config:
        validate_by_name = True  # Enables


class EditHistoryMessageRequestVersion2(BaseModel):
    old_message_id: int = Field(..., alias="oldMessageId")
    new_message_id: int = Field(..., alias="newMessageId")
    new_message: str = Field(..., alias="newMessage")
    chat_id: str = Field(..., alias="chatId")

    class Config:
        validate_by_name = True  # Enables


class ChatDiagnosisMessageEntry(BaseModel):
    is_ai: bool = Field(..., alias="isAi")
    content: str


class ChatDiagnosisRequest(BaseModel):
    chat_id: str = Field(..., alias="chatId")
    diagnosis_message_prompt: str = Field(..., alias="diagnosisMessagePrompt")
    messages: List[ChatDiagnosisMessageEntry] = Field(..., alias="messages")

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
app = modal.App("mentallama-chat-7b")
# Create a Modal Stub (this is your app)
diagnose_app = modal.App("mentallama-diagnose")
# Model name to use
model_name = "klyang/MentaLLaMA-chat-7B"
model_path = "/model"
chat_history_path = "/chat-history"
diagnosis_path = "/diagnosis-history"


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
        model_cache_volume.commit()
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
    model = AutoModelForCausalLM(
        model_path,
        device_map="auto",
        # offload_folder="offload",
        # quantization_config=quantisation_config
    )
    model.eval()

    # Format history for the model
    system_prompt = "You are a helpful, emotionally aware Assistant. Always respond with empathy based on the emotion. Acknowledge their emotional state briefly if it's relevant, then answer their question clearly and factually. If the user is angry or upset, remain calm and polite, but always answer their question."
    # Format history for the model
    chat_input = (
        system_prompt
        + "".join(
            f"{turn['role']}: {turn['content']}\n" for turn in conversation_history
        )
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
def generate_response_version2(message: str, message_id: int, chat_id: str):
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

    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            try:
                conversation_history: list[dict] = json.load(f)

            except json.JSONDecodeError:
                conversation_history: list[dict] = []
    else:
        conversation_history: list[dict] = []

    conversation_history.append(
        {"role": "user", "content": message, "messageId": message_id}
    )

    # endregion

    # Format history for the model
    system_prompt = "You are a helpful, emotionally aware Assistant. Always respond with empathy based on the emotion. Acknowledge their emotional state briefly if it's relevant, then answer their question clearly and factually. If the user is angry or upset, remain calm and polite, but always answer their question. After answering, ask a thoughtful follow-up question related to the user's message to keep the conversation going."

    # Format history for the model
    chat_input = (
        system_prompt
        + "\n\n"
        + "".join(
            f"{turn['role']}: {turn['content']}\n" for turn in conversation_history
        )
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
    for delimeter in ["\nuser:", "\nReasoning", "\nExplanation", "\n\n"]:
        if delimeter in assistant_response:
            assistant_response = assistant_response.split(delimeter)[0].strip()
            break

    # Clear memory
    del model
    del tokenizer
    torch.cuda.empty_cache()

    conversation_history.append({"role": "assistant", "content": assistant_response})

    # Step 3: Write updated list back to the file
    try:

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(conversation_history, f, indent=4, ensure_ascii=False)
    except PermissionError:
        raise PermissionError(f"Permission denied while writing file: {chat_id}")

    except Exception as e:
        raise OSError(f"Error writing file {chat_id}: {str(e)}")

    return assistant_response


@app.function(
    image=image,
    gpu="A100",
    timeout=600,
    volumes={model_path: model_cache_volume, chat_history_path: chat_history_volume},
)
def edit_history_message_v2(
    old_message_id: int, new_message_id: int, new_message: str, chat_id: str
):
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

    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                try:
                    conversation_history: list[dict] = json.load(f)

                except json.JSONDecodeError:
                    conversation_history: list[dict] = []
        except PermissionError:
            raise PermissionError(f"Chat history file is locked or in use: {chat_id}")
        except FileNotFoundError:
            raise FileNotFoundError("Chat history file not found: " + chat_id)
        except Exception as e:
            raise OSError(f"Unexpected error accessing file {chat_id}: {str(e)}")
    else:
        raise FileNotFoundError("Chat history file not found: " + chat_id)

    for i, message in enumerate(conversation_history):

        if message.get("messageId") == old_message_id:
            conversation_history = conversation_history[:i]
            break

    conversation_history.append(
        {"role": "user", "content": new_message, "messageId": new_message_id}
    )

    # endregion

    # Format history for the model
    system_prompt = "You are a helpful, emotionally aware Assistant. Always respond with empathy based on the emotion. Acknowledge their emotional state briefly if it's relevant, then answer their question clearly and factually. If the user is angry or upset, remain calm and polite, but always answer their question."

    # Format history for the model
    chat_input = (
        system_prompt
        + "".join(
            f"{turn['role']}: {turn['content']}\n" for turn in conversation_history
        )
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

    for delimeter in ["\nuser:", "\nReasoning", "\nExplanation", "\n\n"]:
        if delimeter in assistant_response:
            assistant_response = assistant_response.split(delimeter)[0].strip()
            break
    # Clear memory
    del model
    del tokenizer
    torch.cuda.empty_cache()

    conversation_history.append({"role": "assistant", "content": assistant_response})

    # Step 3: Write updated list back to the file
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(conversation_history, f, indent=4, ensure_ascii=False)

    except PermissionError:
        raise PermissionError(f"Permission denied while writing file: {chat_id}")

    except Exception as e:
        raise OSError(f"Error writing file {chat_id}: {str(e)}")

    return assistant_response


@app.function(
    image=image,
    gpu="A100",
    timeout=600,
    volumes={model_path: model_cache_volume, chat_history_path: chat_history_volume},
)
def delete_history_v2(chat_id: str):

    filename = f"{chat_history_path}/{chat_id}.json"

    if os.path.exists(filename):
        try:
            os.remove(filename)
        except FileNotFoundError:
            # File was deleted by something else between exists() and remove()
            raise FileNotFoundError(
                f"Chat history file '{chat_id}' was already deleted."
            )
        except PermissionError as e:
            raise PermissionError(
                f"Chat history file '{chat_id}' is locked or in use: {e}"
            )
        except Exception as e:
            raise OSError(f"Unexpected error accessing file {chat_id}: {str(e)}")

    else:
        raise FileNotFoundError(f"Chat history file not found: {chat_id}")


@app.function(
    image=image,
    gpu="A100",
    timeout=600,
    volumes={
        model_path: model_cache_volume,
        chat_history_path: chat_history_volume,
        diagnosis_path: diagnosis_volume,
    },
)
def generate_diagnosis(
    chat_id: str,
    diagnosis_message_prompt: str,
    messages: list[ChatDiagnosisMessageEntry],
):
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

    diagnosis_filename = f"{diagnosis_path}/{chat_id}.json"

    conversation_history: list[dict] = [
        {
            "role": "assistant" if message_entry.is_ai else "user",
            "content": message_entry.content,
        }
        for message_entry in messages
    ]
    conversation_context = list(conversation_history)
    conversation_history.append({"role": "user", "content": diagnosis_message_prompt})
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

    response_parts = response.split(",", maxsplit=1)
    response_status = response_parts[0].strip()     # "False"
    response_part = response_parts[1].strip()  # The JSON string
    response_json_data = json.loads(response_part)

    

    # Clear memory
    del model
    del tokenizer
    torch.cuda.empty_cache()

    diagnosis_entry = {
        "chat_id": chat_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "diagnosis_message_prompt": diagnosis_message_prompt,
        "diagnosis_response": assistant_response,
        "conversation_context": conversation_context,  # Original chat history without diagnosis interaction
    }

    # Load existing diagnoses for this chat_id if they exist
    if os.path.exists(diagnosis_filename):
        with open(diagnosis_filename, "r", encoding="utf-8") as f:
            try:
                diagnoses_history = json.load(f)
                if not isinstance(diagnoses_history, list):
                    diagnoses_history = [
                        diagnoses_history
                    ]  # Convert old format to list
            except json.JSONDecodeError:
                diagnoses_history = []
    else:
        diagnoses_history = []

    # Add new diagnosis entry
    diagnoses_history.append(diagnosis_entry)

    # Save updated diagnoses to diagnosis volume
    with open(diagnosis_filename, "w", encoding="utf-8") as f:
        json.dump(diagnoses_history, f, indent=4, ensure_ascii=False)

    return response_json_data


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

    @fastapi_app.post("/v2", response_model=ChatResponse)
    async def chat(request: ChatRequestVersion2):

        try:
            response = generate_response_version2.remote(
                request.message, request.message_id, request.chat_id
            )
            return ChatResponse(response=response)

        except PermissionError as e:

            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error generating response: {str(e)}"
            )

    @fastapi_app.post("/edit-history-message-v2", response_model=ChatResponse)
    async def chat_endpoint(request: EditHistoryMessageRequestVersion2):

        try:
            response = edit_history_message_v2.remote(
                request.old_message_id,
                request.new_message_id,
                request.new_message,
                request.chat_id,
            )
            return ChatResponse(response=response)

        except (FileNotFoundError, PermissionError) as e:

            raise HTTPException(status_code=404, detail=str(e))

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error generating response: {str(e)}"
            )

    @fastapi_app.post("/delete-history-v2", status_code=204)
    async def chat(request: DeleteChatRequestVersion2):
        try:
            response = delete_history_v2.remote(
                request.chat_id,
            )
            return Response(status_code=204)

        except (FileNotFoundError, PermissionError) as e:

            raise HTTPException(status_code=404, detail=str(e))

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error generating response: {str(e)}"
            )

    @fastapi_app.post("/get-diagnosis", response_model=ChatResponse)
    async def get_diagnosis(
        request: ChatDiagnosisRequest, api_key: str = Depends(get_api_key)
    ):
        try:
            response = generate_diagnosis.remote(
                request.chat_id, request.diagnosis_message_prompt, request.messages
            )
            return ChatResponse(response=response)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error generating response: {str(e)}"
            )

    # @fastapi_app.post("/", response_model=ChatResponse)
    # async def chat(request: ChatRequest):
    #     try:
    #         response = generate_response.remote(request.history)
    #         return ChatResponse(response=response)
    #     except Exception as e:
    #         raise HTTPException(
    #             status_code=500, detail=f"Error generating response: {str(e)}"
    #         )
    return fastapi_app

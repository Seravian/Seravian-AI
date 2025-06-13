import json
import threading
from typing import List, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field
import modal
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from fastapi import Depends, FastAPI, File, Form, Response, UploadFile, HTTPException
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
import os
import datetime
import time
from pythonjsonlogger import jsonlogger
import logging


# region logging

LOG_FILE_PATH = "/logs/mentallama7b.json"  # Use Modal volume mount path

# Ensure the logs directory exists (defensive check)
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)


class UTCFormatter(jsonlogger.JsonFormatter):
    converter = time.gmtime


# Persistent model cache volume

log_volume = modal.Volume.from_name("log-volume", create_if_missing=True)

formatter = UTCFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")

file_handler = logging.FileHandler(LOG_FILE_PATH, mode="a")
file_handler.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
# endregion

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
    chat_diagnosis_id: int = Field(..., alias="chatDiagnosisId")
    messages: List[ChatDiagnosisMessageEntry] = Field(..., alias="messages")

    class Config:
        validate_by_name = True  # Enables


def to_camel(string: str) -> str:
    parts = string.split("_")
    return parts[0] + "".join(word.capitalize() for word in parts[1:])


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class ChatDiagnosisResponse(CamelModel):
    chat_id: str
    diagnosis_message_prompt: str
    is_succeeded: bool
    diagnosed_problem: Optional[str]
    reasoning: Optional[str]
    prescription: Optional[list[str]]
    failure_reason: Optional[str]


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
    .pip_install("sentencepiece")
    .pip_install("python-json-logger")  # Install separately after build tools
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
    chat_diagnosis_id: int,
    messages: list[ChatDiagnosisMessageEntry],
) -> ChatDiagnosisResponse:
    """
    Generate a response based on the conversation history and user message.
    """

    diagnosis_message_prompt = """You are a mental health assistant. Analyze all the messages in this conversation. Determine whether the user may be suffering from any identifiable mental health issues based on the content and tone of the messages.

If you identify a problem, return your answer in the following JSON format:

{
  "Diagnosed problem": "<Clearly state the mental health issue in no more than 20 words, e.g., Generalized Anxiety Disorder>",
  "Reasoning": "<Explain why you reached this conclusion, based on message patterns or content>",
  "Activities to help with dealing with this problem": ["<List 2–3 simple, practical suggestions and exercises tailored to the issue>"]
}

If you cannot confidently identify a problem, return your answer in this fallback JSON format:

{
  "Diagnose failure reason": "<Clearly explain why no diagnosis could be made (e.g., not enough information, unclear patterns) if there are more reasons on why diagnosis couldn't be made state them clearly.>"
}

IMPORTANT: You must NEVER suggest or prescribe any type of medication. Your role is strictly limited to observational analysis and practical, non-medical suggestions.

Only output one of these two JSON objects, and nothing else."""
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

    response_parts = assistant_response.split(",", maxsplit=1)

    if len(response_parts) != 2:
        # handle when the model fails to generate a response  don't start with {True or False },
        # log that the model generated a response that doesn't match the expected format that don't have one comma
        logger.error(
            f"Model generated a response that doesn't have one comma",
            extra={
                "chat_id": chat_id,
                "diagnosis_id": chat_diagnosis_id,
                "response": response,
            },
        )
        raise Exception(
            f"Model generated a response that doesn't have one comma at `boolean, response` : {response}"
        )

    string_status = response_parts[0].strip()  # "False"
    response_part = response_parts[1].strip()  # The JSON string
    if string_status.lower() != "true" and string_status.lower() != "false":
        # handle when the model fails to generate a response the  part before the comma is not True or False
        logger.error(
            f"Model generated a response that have one comma atleast but the part before the comma is not True or False",
            extra={
                "chat_id": chat_id,
                "diagnosis_id": chat_diagnosis_id,
                "response": response,
            },
        )
        raise Exception(
            f"Model generated a response that have one comma atleast but the part before the comma is not True or False : {response}"
        )

    try:
        response_json_data: dict = json.loads(response_part)
    except json.JSONDecodeError:
        # handle when the model fails to generate a response the  part after the comma is not valid json
        logger.error(
            f"Model generated a response that the part after the comma is not valid json format",
            extra={
                "chat_id": chat_id,
                "diagnosis_id": chat_diagnosis_id,
                "response": response,
            },
        )
        raise Exception(
            f"Model generated a response that the part after the comma is not valid json format : {response}"
        )

    if string_status.lower() == "True":
        # check if the response part match the provided provided JSON format when diagnosis is successful
        if (
            "Diagnosed problem" in response_json_data
            and "Reasoning" in response_json_data
            and "Activities to help with dealing with this problem"
            in response_json_data
            and len(response_json_data) == 3
        ):
            # check that each is not None and not whitespace and list is not None and not empty and also its items are not None and not whitespace
            # and the value of "Activities to help with dealing with this problem"  is list of str
            if (
                response_json_data["Diagnosed problem"]
                and response_json_data["Reasoning"]
                and response_json_data["Diagnosed problem"].strip()
                and response_json_data["Reasoning"].strip()
                and isinstance(
                    response_json_data[
                        "Activities to help with dealing with this problem"
                    ],
                    list,
                )
                and response_json_data[
                    "Activities to help with dealing with this problem"
                ]
                and not all(
                    isinstance(item, str) and item and item.strip()
                    for item in response_json_data[
                        "Activities to help with dealing with this problem"
                    ]
                )
            ):

                # return the response
                return ChatDiagnosisResponse(
                    chat_id=chat_id,
                    diagnosis_message_prompt=diagnosis_message_prompt,
                    is_succeeded=True,
                    diagnosed_problem=response_json_data["Diagnosed problem"],
                    reasoning=response_json_data["Reasoning"],
                    prescription=response_json_data[
                        "Activities to help with dealing with this problem"
                    ],
                    failure_reason=None,
                )
            else:
                logger.error(
                    "Model generated a response that doesn't match the provided provided JSON format when diagnosis is successful",
                    extra={
                        "chat_id": chat_id,
                        "diagnosis_id": chat_diagnosis_id,
                        "response": response,
                    },
                )
                raise Exception(
                    "Model generated a response that doesn't match the provided provided JSON format when diagnosis is successful"
                )
    else:
        # check if the response part match the provided provided JSON format when diagnosis is failed
        if (
            "Diagnose failure reason" in response_json_data
            and len(response_json_data) == 1
        ):
            if (
                response_json_data["Diagnose failure reason"]
                and response_json_data["Diagnosed problem"].strip()
            ):
                return ChatDiagnosisResponse(
                    chat_id=chat_id,
                    diagnosis_message_prompt=diagnosis_message_prompt,
                    is_succeeded=False,
                    diagnosed_problem=None,
                    reasoning=None,
                    prescription=None,
                    failure_reason=response_json_data["Diagnose failure reason"],
                )
            else:
                logger.error(
                    "Model generated a response that doesn't match the provided provided JSON format when diagnosis is failed",
                    extra={
                        "chat_id": chat_id,
                        "diagnosis_id": chat_diagnosis_id,
                        "response": response,
                    },
                )
                raise Exception(
                    "Model generated a response that doesn't match the provided provided JSON format when diagnosis is failed"
                )

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

    @fastapi_app.post("/get-diagnosis", response_model=ChatDiagnosisResponse)
    async def get_diagnosis(
        request: ChatDiagnosisRequest, api_key: str = Depends(get_api_key)
    ):
        try:
            response = generate_diagnosis.remote(
                request.chat_id, request.chat_diagnosis_id, request.messages
            )
            return response
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

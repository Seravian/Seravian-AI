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

LOG_FILE_PATH = "/logs/mentallama13b.json"  # Use Modal volume mount path

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
    diagnosed_problem: Optional[str] = None  # Made optional with default None
    reasoning: Optional[str] = None  # Made optional with default None
    prescription: Optional[list[str]] = None  # Made optional with default None
    failure_reason: Optional[str] = None


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
app = modal.App("mentallama-chat-13b")
# Create a Modal Stub (this is your app)
diagnose_app = modal.App("mentallama-diagnose")
# Model name to use
model_name = "klyang/MentaLLaMA-chat-13B"
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
    Generate a diagnosis based on the conversation history.
    """

    diagnosis_message_prompt = """You are a mental health assistant. Analyze all the messages in this conversation. Determine whether I may be suffering from any identifiable mental health issues based on the content and tone of the messages.

You MUST respond with ONLY a valid JSON object. Do not include any additional text, explanations, or apologies outside of the JSON.

If you identify a problem, return ONLY this JSON:

{
  "Diagnosed problem": "<Clearly state the mental health issue in no more than 20 words, e.g., Generalized Anxiety Disorder>",
  "Reasoning": "<Explain why you reached this conclusion, based on message patterns or content>",
  "Activities to help with dealing with this problem": ["<List 3 simple, practical suggestions and exercises tailored to the issue>"]
}

If you cannot confidently identify a problem, return ONLY this JSON:

{
  "Diagnose failure reason": "<Clearly explain why no diagnosis could be made (e.g., not enough information, unclear patterns)>"
}

CRITICAL: You must NEVER suggest or prescribe any type of medication. Your role is strictly limited to observational analysis and practical, non-medical suggestions.

Output ONLY the JSON object, nothing else."""

    try:
        # Load tokenizer and model from volume
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
        )
        model.eval()

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

        # Format history for the model
        chat_input = (
            "".join(f"{turn['role']}: {turn['content']}\n" for turn in conversation_history)
            + "assistant:"
        )

        # Tokenize and generate response
        inputs = tokenizer(chat_input, return_tensors="pt", truncation=True).to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,  # Significantly increased to ensure complete JSON
            temperature=0.2,  # Even lower temperature for more consistent JSON
            top_p=0.8,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        response: str = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract assistant's response
        assistant_response = response.split("assistant:")[-1].strip()

        # Clear memory first to free up resources
        del model
        del tokenizer
        torch.cuda.empty_cache()

        # Enhanced JSON extraction and cleaning
        json_response = assistant_response.strip()
        
        # Log the raw response for debugging
        logger.info(
            f"Raw model response for diagnosis",
            extra={
                "chat_id": chat_id,
                "diagnosis_id": chat_diagnosis_id,
                "raw_response": json_response,  # Log full response for debugging
                "response_length": len(json_response),
            },
        )
        
        # More aggressive JSON cleaning and extraction
        if "{" in json_response:
            start_idx = json_response.find("{")
            json_response = json_response[start_idx:]
            
            # Handle potential issues with incomplete JSON
            # Check if we have proper structure
            if json_response.count('"') % 2 != 0:
                # Odd number of quotes - likely truncated
                logger.warning(f"Detected odd number of quotes, attempting to fix")
                # Try to complete the JSON structure
                if not json_response.endswith('"'):
                    json_response += '"'
                
                # Check if we need closing braces/brackets
                open_braces = json_response.count("{")
                close_braces = json_response.count("}")
                open_brackets = json_response.count("[")
                close_brackets = json_response.count("]")
                
                # Add missing closing characters
                if open_brackets > close_brackets:
                    json_response += "]" * (open_brackets - close_brackets)
                if open_braces > close_braces:
                    json_response += "}" * (open_braces - close_braces)
            
        # Clean up common formatting issues
        json_response = json_response.replace('"', '"').replace('"', '"')
        json_response = json_response.replace(''', "'").replace(''', "'")
        
        # Additional cleaning for potential line breaks or formatting issues
        json_response = json_response.strip()
        
        # Parse the JSON response with multiple attempts
        response_json_data = None
        parsing_attempts = []
        
        # Attempt 1: Direct parsing
        try:
            response_json_data = json.loads(json_response)
            logger.info(f"JSON parsed successfully on first attempt")
        except json.JSONDecodeError as e:
            parsing_attempts.append(f"Direct parse failed: {str(e)}")
            
            # Attempt 2: Try to fix common issues
            try:
                # Fix potential issues with escaped characters
                fixed_response = json_response.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                response_json_data = json.loads(fixed_response)
                logger.info(f"JSON parsed successfully after escape character fix")
            except json.JSONDecodeError as e2:
                parsing_attempts.append(f"Escape fix failed: {str(e2)}")
                
                # Attempt 3: Try with regex to extract valid JSON parts
                try:
                    import re
                    # Look for complete key-value pairs
                    json_pattern = r'{\s*"[^"]*"\s*:\s*"[^"]*"(?:\s*,\s*"[^"]*"\s*:\s*(?:"[^"]*"|\[[^\]]*\]))*\s*}'
                    matches = re.search(json_pattern, json_response, re.DOTALL)
                    if matches:
                        clean_json = matches.group(0)
                        response_json_data = json.loads(clean_json)
                        logger.info(f"JSON parsed successfully using regex extraction")
                    else:
                        raise json.JSONDecodeError("No valid JSON pattern found", json_response, 0)
                except (json.JSONDecodeError, ImportError) as e3:
                    parsing_attempts.append(f"Regex extraction failed: {str(e3)}")
                    
                    # Final attempt: Manual reconstruction for known structure
                    try:
                        # Try to manually extract the expected fields
                        manual_json = {}
                        
                        # Look for "Diagnosed problem"
                        if '"Diagnosed problem"' in json_response:
                            problem_match = re.search(r'"Diagnosed problem"\s*:\s*"([^"]*)"', json_response)
                            if problem_match:
                                manual_json["Diagnosed problem"] = problem_match.group(1)
                        
                        # Look for "Reasoning"
                        if '"Reasoning"' in json_response:
                            reasoning_match = re.search(r'"Reasoning"\s*:\s*"([^"]*)"', json_response, re.DOTALL)
                            if reasoning_match:
                                manual_json["Reasoning"] = reasoning_match.group(1)
                        
                        # Look for "Activities" array
                        if '"Activities to help with dealing with this problem"' in json_response:
                            activities_match = re.search(r'"Activities to help with dealing with this problem"\s*:\s*\[([^\]]*)\]', json_response, re.DOTALL)
                            if activities_match:
                                activities_str = activities_match.group(1)
                                # Extract individual activities
                                activity_items = re.findall(r'"([^"]*)"', activities_str)
                                manual_json["Activities to help with dealing with this problem"] = activity_items
                        
                        # Look for failure reason
                        if '"Diagnose failure reason"' in json_response:
                            failure_match = re.search(r'"Diagnose failure reason"\s*:\s*"([^"]*)"', json_response, re.DOTALL)
                            if failure_match:
                                manual_json["Diagnose failure reason"] = failure_match.group(1)
                        
                        if manual_json:
                            response_json_data = manual_json
                            logger.info(f"JSON reconstructed manually with keys: {list(manual_json.keys())}")
                        else:
                            raise ValueError("Manual reconstruction failed")
                            
                    except Exception as e4:
                        parsing_attempts.append(f"Manual reconstruction failed: {str(e4)}")
        
        # If all parsing attempts failed
        if response_json_data is None:
            logger.error(
                f"All JSON parsing attempts failed",
                extra={
                    "chat_id": chat_id,
                    "diagnosis_id": chat_diagnosis_id,
                    "response": json_response,
                    "attempts": parsing_attempts,
                },
            )
            return ChatDiagnosisResponse(
                chat_id=chat_id,
                diagnosis_message_prompt=diagnosis_message_prompt,
                is_succeeded=False,
                failure_reason=json_response
            )
        
        logger.info(
            f"Successfully parsed JSON for diagnosis",
            extra={
                "chat_id": chat_id,
                "diagnosis_id": chat_diagnosis_id,
                "parsed_keys": list(response_json_data.keys()),
            },
        )

        # Validate and process successful diagnosis
        if all(key in response_json_data for key in ["Diagnosed problem", "Reasoning", "Activities to help with dealing with this problem"]):
            diagnosed_problem = response_json_data.get("Diagnosed problem", "").strip()
            reasoning = response_json_data.get("Reasoning", "").strip()
            activities = response_json_data.get("Activities to help with dealing with this problem", [])
            
            # Validate the data
            if not diagnosed_problem:
                return ChatDiagnosisResponse(
                    chat_id=chat_id,
                    diagnosis_message_prompt=diagnosis_message_prompt,
                    is_succeeded=False,
                    failure_reason="No Diagnosed Problem."
                )
            
            if not reasoning:
                return ChatDiagnosisResponse(
                    chat_id=chat_id,
                    diagnosis_message_prompt=diagnosis_message_prompt,
                    is_succeeded=False,
                    failure_reason="No Reasoning Found."
                )
            
            if not isinstance(activities, list) or not activities:
                return ChatDiagnosisResponse(
                    chat_id=chat_id,
                    diagnosis_message_prompt=diagnosis_message_prompt,
                    is_succeeded=False,
                    failure_reason="No Activities  or excercises could be prescribed."
                )
            
            # Ensure all activities are valid strings
            valid_activities = [str(activity).strip() for activity in activities if str(activity).strip()]
            if not valid_activities:
                return ChatDiagnosisResponse(
                    chat_id=chat_id,
                    diagnosis_message_prompt=diagnosis_message_prompt,
                    is_succeeded=False,
                    failure_reason="No Activities  or excercises could be prescribed."
                )

            # Save successful diagnosis
            diagnosis_entry = {
                "chat_id": chat_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "diagnosis_message_prompt": diagnosis_message_prompt,
                "diagnosis_response": assistant_response,
                "conversation_context": conversation_context,
            }

            try:
                if os.path.exists(diagnosis_filename):
                    with open(diagnosis_filename, "r", encoding="utf-8") as f:
                        try:
                            diagnoses_history = json.load(f)
                            if not isinstance(diagnoses_history, list):
                                diagnoses_history = [diagnoses_history]
                        except json.JSONDecodeError:
                            diagnoses_history = []
                else:
                    diagnoses_history = []

                diagnoses_history.append(diagnosis_entry)

                with open(diagnosis_filename, "w", encoding="utf-8") as f:
                    json.dump(diagnoses_history, f, indent=4, ensure_ascii=False)
            except Exception as save_error:
                logger.error(f"Failed to save diagnosis: {save_error}")

            return ChatDiagnosisResponse(
                chat_id=chat_id,
                diagnosis_message_prompt=diagnosis_message_prompt,
                is_succeeded=True,
                diagnosed_problem=diagnosed_problem,
                reasoning=reasoning,
                prescription=valid_activities,
            )

        # Handle failure case
        elif "Diagnose failure reason" in response_json_data:
            failure_reason = response_json_data.get("Diagnose failure reason", "").strip()
            
            if not failure_reason:
                return ChatDiagnosisResponse(
                    chat_id=chat_id,
                    diagnosis_message_prompt=diagnosis_message_prompt,
                    is_succeeded=False,
                    failure_reason="Failed to provide Failure Reason"
                )
            
            return ChatDiagnosisResponse(
                chat_id=chat_id,
                diagnosis_message_prompt=diagnosis_message_prompt,
                is_succeeded=False,
                failure_reason=failure_reason,
            )

        # Handle unexpected format
        else:
            logger.error(
                "Model generated response with unexpected JSON structure",
                extra={
                    "chat_id": chat_id,
                    "diagnosis_id": chat_diagnosis_id,
                    "response": assistant_response,
                    "json_keys": list(response_json_data.keys()),
                },
            )
            return ChatDiagnosisResponse(
                chat_id=chat_id,
                diagnosis_message_prompt=diagnosis_message_prompt,
                is_succeeded=False,
                failure_reason=f"Unexpected Output."
            )

    except Exception as e:
        logger.error(
            f"Error in generate_diagnosis: {str(e)}",
            extra={
                "chat_id": chat_id,
                "diagnosis_id": chat_diagnosis_id,
                "error": str(e),
            },
        )
        return ChatDiagnosisResponse(
            chat_id=chat_id,
            diagnosis_message_prompt=diagnosis_message_prompt,
            is_succeeded=False,
            failure_reason=f"Internal error: {str(e)}"
        )


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
            return Response(
                status_code=400, content=f"Error generating response: {str(e)}"
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
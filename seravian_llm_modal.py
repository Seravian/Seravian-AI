from typing import List, Dict
from pydantic import BaseModel, ValidationError
import modal
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
from flask import Flask, jsonify,request # type: ignore
class ChatRequest(BaseModel):
    history: List[Dict[str, str]]
    message: str 
# Step 1: Create the image with dependencies
image = (
    modal.Image.debian_slim()
    .apt_install("build-essential", "cmake", "g++", "git")  # Add build tools
    .pip_install("flask")
    .pip_install("torch", "transformers","pydantic","accelerate","BitsandBytes")   # Install these first
    .pip_install("sentencepiece")  # Install separately after build tools
    
)

# Step 2: Create a Modal Stub (this is your app)
app = modal.App("mentallama-chat-7b")

# Step 3: Define the function that will serve the model
@app.function(image=image, gpu="T4", timeout=600)
def generate_llm_response(history, message):
    model_id = "klyang/MentaLLaMA-chat-7B"
    quantisation_config=BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True
    )
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=quantisation_config)
    
    # Format the conversation history and input
    history.append({"role": "user", "content": message})
    prompt = "".join(f"{turn['role']}: {turn['content']}" for turn in history) + "assistant:"
    
    # Tokenize and generate the response
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant's response
    assistant_response = response.split("assistant:")[-1].strip()
    history.append({"role": "assistant", "content": assistant_response})
    return assistant_response

# Define the FastAPI endpoint
@app.function(image=image)
@modal.concurrent(max_inputs=1000)
@modal.wsgi_app()
def seravian_llm():
    flask_app = Flask(__name__)

    @flask_app.route("/", methods=["POST"])
    def chat():
        try:
            data = request.get_json()
            parsed = ChatRequest(**data)  # Pydantic validation
        except ValidationError as e:
            return jsonify({"error": "Invalid input", "details": e.errors()}), 422

        response = generate_llm_response.remote(parsed.history, parsed.message)
        return jsonify({"response": response})

    return flask_app
  
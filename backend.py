import os
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== MODEL SETUP =====
# Allow overriding the model path/name via env vars so we can point to a local OpenLLaMA checkout.
MODEL_NAME = (
    os.environ.get("THERAPY_GARDEN_MODEL")
    or os.environ.get("OPENLLAMA_PATH")
    or "openlm-research/open_llama_3b_v2"
    or "tanusrich/Mental_Health_Chatbot"
    )
logger.info(f"Loading model: {MODEL_NAME}")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Model loaded on device: {device}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

# ===== FASTAPI APP =====
app = FastAPI(title="Therapy Garden Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== DATA MODELS =====
class HealthRequest(BaseModel):
    pass

class ChatRequest(BaseModel):
    message: str
    coreBeliefs: list[str] = []
    chatHistory: list[dict] = []

# ===== ENDPOINTS =====
@app.get("/api/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "device": str(device)}

@app.post("/api/chat")
def therapy_chat(req: ChatRequest):
    try:
        # Crisis detection
        crisis_keywords = ["suicidal", "self-harm", "self harm", "kill myself", "hurt myself"]
        if any(keyword in req.message.lower() for keyword in crisis_keywords):
            return {
                "reply": (
                    "I notice you mentioned something serious. This is beyond what I can help with.\n\n"
                    "Please reach out immediately:\n"
                    "- National Suicide Prevention Lifeline: 988 (call/text)\n"
                    "- Crisis Text Line: Text HOME to 741741\n"
                    "- Emergency Services: 911\n\n"
                    "Your safety matters. Please talk to someone who can help right now."
                ),
                "isCrisis": True
            }

        # Build system prompt with core beliefs
        core_beliefs_text = "\n".join(f"- {b}" for b in req.coreBeliefs[:7]) if req.coreBeliefs else ""
        
        system_prompt = (
            "You are a supportive, non-clinical mental health journaling companion.\n"
            "You are NOT a therapist and cannot give medical advice, diagnosis, or crisis guidance.\n"
            "For serious concerns, encourage the person to speak with their therapist or a professional.\n\n"
        )
        
        if core_beliefs_text:
            system_prompt += f"Core Beliefs:\n{core_beliefs_text}\n\n"
        
        system_prompt += "Keep responses warm, reflective, and brief (2-3 sentences typical). Ask gentle follow-up questions when appropriate.\n\n"

        # Build conversation history
        history_text = ""
        for msg in req.chatHistory[-6:]:  # Last 6 messages
            role = msg.get("role", "user")
            content = msg.get("message", "")
            if role == "user":
                history_text += f"User: {content}\n"
            else:
                history_text += f"Assistant: {content}\n"

        # Final prompt
        prompt = f"{system_prompt}{history_text}User: {req.message}\nAssistant:"

        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
            )

        full_response = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        if "Assistant:" in full_response:
            reply = full_response.split("Assistant:")[-1].strip()
        else:
            reply = full_response.strip()

        # Truncate if too long
        if len(reply) > 300:
            reply = reply[:300] + "..."

        return {
            "reply": reply,
            "isCrisis": False
        }

    except Exception as e:
        logger.error(f"Error in therapy_chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== RUN SERVER =====
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

import torch
import whisper
import tempfile
import os
import soundfile as sf
import numpy as np
import chromadb
from llama_cpp import Llama
import time
from datetime import datetime, timedelta

from src.config import MODEL_PATH, WHISPER_MODEL
from src.utils import _parse_live_stream

# --- 1. INITIALIZE DATABASES & MODELS ---

print("Initializing Local Ambient Memory (ChromaDB)...")
chroma_client = chromadb.PersistentClient(path="./vanisetu_vectordb")
# This creates a local SQLite database in your repo to store memories
ambient_collection = chroma_client.get_or_create_collection(name="ambient_logs")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Whisper on {DEVICE}...")
whisper_model = whisper.load_model(WHISPER_MODEL, device=DEVICE)

print(f"Loading Gemma 2 2B GGUF on GPU VRAM...")
llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=-1, 
    n_ctx=2048,
    verbose=False 
)
print("System Ready.")

# --- 2. MEMORY MANAGEMENT (RAG) ---


def log_ambient_memory(text: str):
    """Saves a memory with an exact Unix timestamp."""
    if not text or len(text.strip()) < 5:
        return
    
    current_timestamp = time.time()
    doc_id = f"log_{int(current_timestamp * 1000)}"
    
    # We now inject the exact time into the metadata
    ambient_collection.add(
        documents=[text],
        metadatas=[{"timestamp": current_timestamp, "type": "ambient_audio"}],
        ids=[doc_id]
    )
    print(f"💾 Saved to Vector DB with timestamp: '{text}'")

def log_user_choice(context: str, chosen_text: str):
    """Saves the user's final spoken choice to learn their communication style."""
    if not chosen_text or len(chosen_text.strip()) < 2:
        return
        
    current_timestamp = time.time()
    doc_id = f"pref_{int(current_timestamp * 1000)}"
    
    # We combine what was happening + what the user chose to say
    memory_document = f"When context was '{context}', the user chose to say: '{chosen_text}'"
    
    # Notice the different metadata type!
    ambient_collection.add(
        documents=[memory_document],
        metadatas=[{"timestamp": current_timestamp, "type": "user_spoken"}],
        ids=[doc_id]
    )
    print(f"🧠 Learned User Preference: '{chosen_text}'")

def retrieve_recent_context(query: str, max_hours_old: int = 24) -> str:
    """Retrieves both ambient facts and past user preferences."""
    if ambient_collection.count() == 0:
        return "No recent context."
        
    cutoff_time = time.time() - (max_hours_old * 3600)
    
    # Query ChromaDB for the top 3 most relevant memories
    results = ambient_collection.query(
        query_texts=[query],
        n_results=3,
        where={"timestamp": {"$gte": cutoff_time}} 
    )
    
    if not results or not results['documents'] or not results['documents'][0]:
        return "No relevant context."
        
    # We pass the raw memories directly to the prompt
    return " | ".join(results['documents'][0])

def retrieve_context(query: str) -> str:
    """Searches the Vector DB for mathematically similar past events."""
    if ambient_collection.count() == 0:
        return "No recent ambient context."
        
    results = ambient_collection.query(
        query_texts=[query],
        n_results=2 # Get the top 2 most relevant past memories
    )
    
    if results and results['documents'] and results['documents'][0]:
        return " | ".join(results['documents'][0])
    return "No relevant context found."

# --- 3. THE GENERATION ENGINE ---

def _transcribe_audio(audio_tuple):
    sample_rate, audio_array = audio_tuple
    audio_array = audio_array.astype(np.float32)
    if audio_array.max() > 1.0:
        audio_array /= 32768.0

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio_array, sample_rate)
        tmp_path = tmp.name

    # result = whisper_model.transcribe(tmp_path, fp16=False)
    result = whisper_model.transcribe(tmp_path, fp16=False, condition_on_previous_text=False, temperature=0.0)
    os.remove(tmp_path)
    return result["text"]

def process_input_stream(audio_tuple, text_input: str, conversation_history: list, modifier: str, target_language: str):
    if conversation_history is None:
        conversation_history = []

    is_audio_mode = audio_tuple is not None

    # Listen
    user_text = text_input
    if is_audio_mode:
        user_text = _transcribe_audio(audio_tuple)
        if not user_text.strip():
            user_text = "[Inaudible]"

    # RAG: Retrieve context from ChromaDB based on what was just heard!
    ambient_context = retrieve_context(user_text)

    modifier_note = f" (Instruction: {modifier})" if modifier and modifier != "None" else ""
    
    history_prompt = ""
    for turn in conversation_history[-3:]: # Keep tight for SLMs
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if isinstance(content, list): 
            content = content[0].get("text", "")
        history_prompt += f"<start_of_turn>{role}\n{content}<end_of_turn>\n"

    # FEW-SHOT PROMPTING: We explicitly show Gemma how to format the output.
    system_prompt = (
        f"You are VaniSetu, an AAC voice assistant. Generate exactly 3 highly expressive conversational options. "
        f"Do NOT write conversational filler. Output ONLY the numbered list.\n"
        f"Recent Ambient Memory/Context: {ambient_context}\n\n"
        f"--- EXAMPLE ---\n"
        f"User: 'do you want to come to the movie?'\n"
        f"Context: 'mom asked to come home after school'\n"
        f"1. Mom asked me to come home after school, you guys carry on! Thanks anyway.\n"
        f"2. I'd love to, but I have to head straight home today.\n"
        f"3. Maybe next time! Mom needs me at home.\n"
        f"--- END EXAMPLE ---"
    )
    
    prompt = (
        f"<start_of_turn>user\n{system_prompt}<end_of_turn>\n"
        f"{history_prompt}"
        f"<start_of_turn>user\nNew Input: '{user_text}'.{modifier_note}\nGenerate the 3 options.<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

    response_generator = llm(
        prompt,
        max_tokens=150,
        temperature=0.65,
        stop=["<end_of_turn>"],
        stream=True 
    )

    full_text = ""
    
    for chunk in response_generator:
        full_text += chunk["choices"][0]["text"]
        opt1, opt2, opt3 = _parse_live_stream(full_text)
        yield user_text, opt1, opt2, opt3, conversation_history

    user_content = user_text if not is_audio_mode else f"[Audio] {user_text}"
    updated_history = conversation_history + [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": full_text.strip()},
    ]
    
    yield user_text, opt1, opt2, opt3, updated_history
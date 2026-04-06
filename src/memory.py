import json
import os
import re
import torch
from src.config import MEMORY_FILE
from src.utils import _strip_chat_template

def init_memory():
    if not os.path.exists(MEMORY_FILE):
        initial_memory = {
            "user_name": "User", 
            "context": "Using AAC to assist with daily communication.",
            "preferences": [
                "Prefers direct, practical communication.",
                "Languages known: English, Telugu, Hindi.",
            ],
        }
        with open(MEMORY_FILE, "w") as f:
            json.dump(initial_memory, f, indent=2)

def get_memory_string():
    init_memory()
    with open(MEMORY_FILE, "r") as f:
        mem = json.load(f)
    prefs = "\n".join(f"  - {p}" for p in mem["preferences"])
    return f"User: {mem['user_name']}\nContext: {mem['context']}\nPreferences:\n{prefs}"

def save_new_rule(transcript, chosen_response):
    if not transcript or not chosen_response:
        return "Status: ⚠️ Nothing to save."
    with open(MEMORY_FILE, "r") as f:
        mem = json.load(f)
    rule = f"When discussing '{transcript[:40]}', preferred: '{chosen_response[:60]}'"
    mem["preferences"].append(rule)
    with open(MEMORY_FILE, "w") as f:
        json.dump(mem, f, indent=2)
    return "Status: 💾 Preference saved to long-term memory."

def distill_and_save_session(full_transcript):
    if not full_transcript or len(full_transcript.split()) < 10:
        return "Status: ⏸️ Transcript too short to distill."

    extraction_prompt = (
        "Extract ONLY permanent facts about the user's preferences, health, "
        "relationships, or work from this transcript. Ignore small talk.\n"
        f"Transcript: '{full_transcript}'\n"
        "If none, output exactly: NONE\n"
        "If found, output a numbered list of facts."
    )
    
    # Import the local Gemma loader
    from src.engine import _generate_with_gemma
    prompt = f"<start_of_turn>user\n{extraction_prompt}<end_of_turn>\n<start_of_turn>model\n"
    
    response = _generate_with_gemma(prompt)

    if "NONE" in response.upper():
        return "Status: ⏸️ No new permanent facts found."

    facts = re.findall(r"\d+\.\s*(.*)", response)
    if facts:
        with open(MEMORY_FILE, "r") as f:
            mem = json.load(f)
        added = 0
        for fact in facts:
            clean = fact.replace("**", "").strip()
            if clean and clean not in mem["preferences"]:
                mem["preferences"].append(clean)
                added += 1
        with open(MEMORY_FILE, "w") as f:
            json.dump(mem, f, indent=2)
        return f"Status: ✅ Auto-saved {added} new facts from session."

    return "Status: ⏸️ Memory up to date."
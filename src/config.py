import os

MODEL_PATH = "models/gemma-2-2b-it-Q4_K_M.gguf" # using 4 bit int quantized model to make it work in less memory with a tradoff in accuracy!
WHISPER_MODEL = "tiny" # using whisper instead of using llms that are capable of handling audio (Gemma4:[2B,4B]) to make it work in less capable hardware!
MEMORY_FILE = "vanisetu_memory.json"

# Initial support is for these languages and will be extended once the capabilities grow promising!
VOICE_MAP = {
    "English": "en-IN-PrabhatNeural",
    "Telugu":  "te-IN-MohanNeural",
    "Hindi":   "hi-IN-MadhurNeural",
}
import os
import time
import wave
import queue
import threading
import pygame
import glob
from piper import PiperVoice

# 1. Setup the dedicated storage folder
AUDIO_CACHE_DIR = "audio_cache"
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True) # Creates the folder if it doesn't exist

# Initialize Hardware Audio
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
pygame.mixer.init()

# Setup Piper Offline TTS
VOICE_MODEL = "models/en_US-lessac-medium.onnx"
print(f"Loading Piper TTS Voice: {VOICE_MODEL} on CPU...")
try:
    voice = PiperVoice.load(VOICE_MODEL)
except Exception as e:
    print(f"⚠️ Error loading Piper voice. Ensure .onnx and .json files are in the models/ folder. ({e})")

audio_queue = queue.Queue()

def _audio_worker():
    """Background thread that plays local audio seamlessly."""
    while True:
        audio_file = audio_queue.get()
        if audio_file:
            try:
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.05)
            except Exception as e:
                print(f"Audio playback error: {e}")
        audio_queue.task_done()

threading.Thread(target=_audio_worker, daemon=True).start()

def cleanup_old_audio(max_age_minutes=10):
    """Quietly deletes audio files older than the specified time to prevent disk bloat."""
    now = time.time()
    # Find all .wav files in our dedicated folder
    files = glob.glob(os.path.join(AUDIO_CACHE_DIR, "*.wav"))
    
    for f in files:
        try:
            # Check how old the file is (in seconds)
            file_age = now - os.path.getmtime(f)
            if file_age > (max_age_minutes * 60):
                os.remove(f)
        except Exception:
            # If the file is currently being played by Pygame, Windows will block deletion.
            # We just silently pass and it will get deleted the next time this runs!
            pass

def stream_to_speech(text: str, target_language: str = "English"):
    """Synthesizes text to a local WAV file, cleans old files, and plays it."""
    if not text or len(text.strip()) < 2:
        return
        
    # 2. Run the garbage collector BEFORE generating new audio
    cleanup_old_audio(max_age_minutes=10)
    
    # 3. Save the new file securely inside the cache folder
    output_wav = os.path.join(AUDIO_CACHE_DIR, f"speech_{int(time.time() * 1000)}.wav")
    
    with wave.open(output_wav, 'wb') as wav_file:
        wav_file.setnchannels(1) 
        wav_file.setsampwidth(2) 
        wav_file.setframerate(getattr(voice.config, 'sample_rate', 22050) if hasattr(voice, 'config') else 22050)
        voice.synthesize_wav(text, wav_file)
        
    audio_queue.put(output_wav)

def _parse_live_stream(raw_text):
    """Safely extracts options 1, 2, and 3 from a partially generated text stream."""
    import re
    opts = ["", "", ""]
    lines = re.split(r'\n\d+\.\s*', '\n' + raw_text)
    
    idx = 0
    for line in lines:
        clean = line.replace("Transcript:", "").strip()
        if clean and not clean.startswith("[Inaudible]"):
            if idx < 3:
                opts[idx] = clean
                idx += 1
    return opts[0], opts[1], opts[2]

def _strip_chat_template(text: str) -> str:
    """Removes Gemma chat template tags from text for clean memory saving."""
    if not text:
        return ""
    text = text.replace("<start_of_turn>user\n", "")
    text = text.replace("<start_of_turn>model\n", "")
    text = text.replace("<end_of_turn>\n", "")
    text = text.replace("<end_of_turn>", "")
    return text.strip()
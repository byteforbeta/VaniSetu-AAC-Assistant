import torch
import pyaudio
import numpy as np
import threading
import queue
import pygame  

audio_event_queue = queue.Queue()

def start_background_listener():
    print("Loading Silero VAD (Voice Activity Detection)...")
    vad_model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        trust_repo=True
    )
    _, _, _, VADIterator, _ = utils
    vad_iterator = VADIterator(vad_model, min_silence_duration_ms=700)

    def _listen_loop():
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=512)
        print("🟢 Always-On Mic Active. Waiting for speech...")
        
        audio_buffer = []
        while True:
            # NEW FIX: If the device is currently speaking, ignore the microphone!
            if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
                audio_buffer = [] # Clear out any half-heard words
                vad_iterator.reset_states()
                continue # Skip the rest of the loop until it finishes speaking
                
            chunk = stream.read(512, exception_on_overflow=False)
            audio_int16 = np.frombuffer(chunk, np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            
            speech_dict = vad_iterator(torch.from_numpy(audio_float32))
            
            if speech_dict:
                if 'start' in speech_dict:
                    audio_buffer = [audio_float32]
                elif 'end' in speech_dict:
                    final_audio = np.concatenate(audio_buffer)
                    audio_event_queue.put((16000, final_audio))
                    audio_buffer = []
                    vad_iterator.reset_states()
            elif len(audio_buffer) > 0:
                audio_buffer.append(audio_float32)

    threading.Thread(target=_listen_loop, daemon=True).start()

import torch
import pyaudio
import numpy as np

# Load the microscopic VAD model into CPU RAM
vad_model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False
)
(get_speech_timestamps, _, _, VADIterator, _) = utils

# Create a streaming iterator that listens for silences
vad_iterator = VADIterator(vad_model, min_silence_duration_ms=500)

def listen_continuously():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=512)
    
    print("🎙️ Listening... (Speak naturally, no buttons required)")
    audio_buffer = []
    
    while True:
        # Read a tiny 32ms chunk of audio from the room
        chunk = stream.read(512, exception_on_overflow=False)
        audio_int16 = np.frombuffer(chunk, np.int16)
        audio_float32 = int2float(audio_int16) # Normalize
        
        # Feed it to the VAD
        speech_dict = vad_iterator(torch.from_numpy(audio_float32))
        
        if speech_dict:
            if 'start' in speech_dict:
                print("🗣️ Speech detected! Recording...")
                audio_buffer = [audio_float32]
            
            elif 'end' in speech_dict:
                print("⏸️ Silence detected. Sending to Whisper...")
                # The user stopped talking. Stitch the audio together!
                final_audio = np.concatenate(audio_buffer)
                
                # ---> SEND TO WHISPER ---> SEND TO GEMMA --->
                process_input((16000, final_audio), ...)
                
                # Reset the buffer for the next time they speak
                audio_buffer = []
                vad_iterator.reset_states()
        
        # If we are currently in the middle of a spoken sentence, keep saving the chunks
        elif len(audio_buffer) > 0:
            audio_buffer.append(audio_float32)
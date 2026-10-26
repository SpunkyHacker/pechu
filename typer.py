import time
import queue
import threading
import subprocess
import numpy as np
import sounddevice as sd
from pynput import keyboard
import mlx_whisper

# --- CONFIGURATION ---
SAMPLE_RATE = 16000
CHUNK_DURATION = 2.5  # Seconds per live update while speaking
MODEL_REPO = "mlx-community/whisper-small-mlx"  # Fast & accurate on M-series
TRIGGER_KEY = keyboard.Key.alt_r  # Hold Right Option key to talk

# --- STATE ---
is_recording = False
audio_queue = queue.Queue()
kb_controller = keyboard.Controller()

print("Pre-loading Whisper model into unified memory...")
# Warm up the model with 1 second of silence so the first press is instant
mlx_whisper.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), path_or_hf_repo=MODEL_REPO)
print("✅ Ready! Hold DOWN [Right Option] anywhere and start talking...")


def paste_text(text: str):
    """Injects text instantly at the cursor via macOS clipboard."""
    cleaned = text.strip()
    if not cleaned:
        return
    
    # Add a trailing space so consecutive chunks don't mash together
    payload = cleaned + " "
    
    # Copy to macOS clipboard using pbcopy
    process = subprocess.Popen(['pbcopy'], stdin=subprocess.PIPE)
    process.communicate(payload.encode('utf-8'))
    
    # Simulate Cmd + V
    with kb_controller.pressed(keyboard.Key.cmd):
        kb_controller.tap('v')


def audio_callback(indata, frames, time_info, status):
    """Triggered continuously by sounddevice while mic stream is open."""
    if is_recording:
        audio_queue.put(indata.copy())


def transcription_worker():
    """Background thread that transcribes audio chunks as you speak."""
    buffer = []
    samples_per_chunk = int(SAMPLE_RATE * CHUNK_DURATION)

    while True:
        try:
            # Wait briefly for incoming audio frames
            frame = audio_queue.get(timeout=0.1)
            buffer.append(frame)
            
            current_samples = sum(len(f) for f in buffer)
            
            # If we hit 2.5 seconds of audio while still holding the key, transcribe it live
            if current_samples >= samples_per_chunk:
                audio_data = np.concatenate(buffer, axis=0).flatten()
                buffer = []  # Reset buffer for next words
                
                result = mlx_whisper.transcribe(
                    audio_data,
                    path_or_hf_repo=MODEL_REPO,
                    fp16=True,
                    language="en"
                )
                paste_text(result.get("text", ""))

        except queue.Empty:
            # If key was released and we have leftover audio in the buffer, flush it
            if not is_recording and len(buffer) > 0:
                audio_data = np.concatenate(buffer, axis=0).flatten()
                buffer = []
                
                # Ignore accidental taps shorter than 0.3 seconds
                if len(audio_data) > SAMPLE_RATE * 0.3:
                    result = mlx_whisper.transcribe(
                        audio_data,
                        path_or_hf_repo=MODEL_REPO,
                        fp16=True,
                        language="en"
                    )
                    paste_text(result.get("text", ""))


# Start the background transcription thread
threading.Thread(target=transcription_worker, daemon=True).start()

# Open the microphone stream (stays idle until is_recording is True)
stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', callback=audio_callback)
stream.start()


def on_press(key):
    global is_recording
    if key == TRIGGER_KEY and not is_recording:
        is_recording = True
        print("🎙️ Listening...")


def on_release(key):
    global is_recording
    if key == TRIGGER_KEY and is_recording:
        is_recording = False
        print("⏹️ Stopped.")


# Listen for global hotkeys
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()

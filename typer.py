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
# Using large-v3-turbo: Blazing fast and wildly accurate on M-series unified memory
MODEL_REPO = "mlx-community/whisper-large-v3-turbo"  
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
    """Pops clean text into the macOS clipboard and fires Cmd + V once."""
    if not text:
        return
    
    # Add a trailing space so consecutive voice notes don't slam together
    payload = text + " "
    
    process = subprocess.Popen(['pbcopy'], stdin=subprocess.PIPE)
    process.communicate(payload.encode('utf-8'))
    
    with kb_controller.pressed(keyboard.Key.cmd):
        kb_controller.tap('v')


def audio_callback(indata, frames, time_info, status):
    """Triggered continuously by sounddevice while mic stream is open."""
    if is_recording:
        audio_queue.put(indata.copy())


def transcription_worker():
    global is_recording
    audio_buffer = []
    was_recording = False

    while True:
        try:
            # Grab audio frames as they come in from the microphone stream
            frame = audio_queue.get(timeout=0.05)
            
            # CRITICAL FIX: Only save audio chunks while recording is active
            if is_recording:
                audio_buffer.append(frame)
                was_recording = True
                
        except queue.Empty:
            pass

        # The exact moment you let go of the key (`is_recording` turns False):
        if not is_recording and was_recording:
            was_recording = False
            
            if len(audio_buffer) > 0:
                # Combine all audio chunks into a single NumPy array
                audio_data = np.concatenate(audio_buffer, axis=0).flatten()
                audio_buffer = []  # Clear buffer for next time

                # Ignore accidental micro-taps shorter than 0.4 seconds
                if len(audio_data) > SAMPLE_RATE * 0.4:
                    result = mlx_whisper.transcribe(
                        audio_data,
                        path_or_hf_repo=MODEL_REPO,
                        fp16=True,
                        language="en"
                    )
                    final_text = result.get("text", "").strip()
                    if final_text:
                        paste_text(final_text)


# Start the background transcription thread
threading.Thread(target=transcription_worker, daemon=True).start()

# Open the microphone stream
stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', callback=audio_callback)
stream.start()


def on_press(key):
    global is_recording
    if key == TRIGGER_KEY and not is_recording:
        # Clear any stale audio sitting in the queue from previous runs
        while not audio_queue.empty():
            audio_queue.get_nowait()
            
        is_recording = True
        subprocess.Popen(["afplay", "/System/Library/Sounds/Tink.aiff"])
        print("🎙️ Listening...")


def on_release(key):
    global is_recording
    if key == TRIGGER_KEY and is_recording:
        is_recording = False
        subprocess.Popen(["afplay", "/System/Library/Sounds/Pop.aiff"])
        print("⏹️ Stopped. Transcribing & pasting...")


# Listen for global hotkeys
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()


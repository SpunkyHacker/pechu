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
UPDATE_INTERVAL = 0.5  # Transcribe every 0.5 seconds for instant word appearance
MODEL_REPO = "mlx-community/whisper-small-mlx" 
TRIGGER_KEY = keyboard.Key.alt_r  

# --- STATE ---
is_recording = False
audio_queue = queue.Queue()
kb_controller = keyboard.Controller()

print("Loading model into unified memory...")
mlx_whisper.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), path_or_hf_repo=MODEL_REPO)
print("✅ Ready! Hold DOWN [Right Option] and start talking...")


def paste_text(text: str):
    """Injects text instantly at the cursor via macOS clipboard."""
    if not text:
        return
    
    # Add a trailing space so the next batch of words doesn't stick
    payload = text + " "
    
    # Copy to macOS clipboard using pbcopy
    process = subprocess.Popen(['pbcopy'], stdin=subprocess.PIPE)
    process.communicate(payload.encode('utf-8'))
    
    # Simulate Cmd + V
    with kb_controller.pressed(keyboard.Key.cmd):
        kb_controller.tap('v')


def audio_callback(indata, frames, time_info, status):
    if is_recording:
        audio_queue.put(indata.copy())


def transcription_worker():
    global is_recording
    
    audio_buffer = []
    typed_word_count = 0
    last_process_time = time.time()
    
    while True:
        try:
            # Grab audio frames rapidly
            frame = audio_queue.get(timeout=0.05)
            audio_buffer.append(frame)
        except queue.Empty:
            pass

        # If key is released, reset everything for the next button press
        if not is_recording:
            audio_buffer = []
            typed_word_count = 0
            continue

        current_time = time.time()
        
        # Every 0.5 seconds, transcribe the ENTIRE growing audio buffer
        if current_time - last_process_time >= UPDATE_INTERVAL and len(audio_buffer) > 0:
            last_process_time = current_time
            
            # Combine all audio recorded so far during this key press
            audio_data = np.concatenate(audio_buffer, axis=0).flatten()
            
            # Skip if audio is too short (prevents Whisper hallucinations)
            if len(audio_data) < SAMPLE_RATE * 0.5:
                continue

            # Run the transcription
            result = mlx_whisper.transcribe(
                audio_data, 
                path_or_hf_repo=MODEL_REPO, 
                fp16=True, 
                language="en"
            )
            current_text = result.get("text", "").strip()
            
            # Split the recognized text into a list of words
            current_words = current_text.split()
            
            # If Whisper found new words we haven't typed yet...
            if len(current_words) > typed_word_count:
                # Extract ONLY the new words
                new_words = current_words[typed_word_count:]
                new_text = " ".join(new_words)
                
                paste_text(new_text)
                
                # Update our count so we don't type them again
                typed_word_count = len(current_words)


# Start the background transcription thread
threading.Thread(target=transcription_worker, daemon=True).start()

# Open the microphone stream
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
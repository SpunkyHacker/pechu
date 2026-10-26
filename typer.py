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
UPDATE_INTERVAL = 1  # Seconds per update of the typed text
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





def audio_callback(indata, frames, time_info, status):
    """Triggered continuously by sounddevice while mic stream is open."""
    if is_recording:
        audio_queue.put(indata.copy())
def replace_text(old_text_len: int, new_text: str):
    """Highlights previously typed text and pastes the updated version over it."""
    # 1. Highlight what we typed in the last 0.5 seconds
    if old_text_len > 0:
        with kb_controller.pressed(keyboard.Key.shift):
            for _ in range(old_text_len):
                kb_controller.tap(keyboard.Key.left)
    
    # 2. Paste the newly updated text over it
    if new_text:
        # Add a trailing space so the next sentence starts cleanly
        payload = new_text + " "
        process = subprocess.Popen(['pbcopy'], stdin=subprocess.PIPE)
        process.communicate(payload.encode('utf-8'))
        
        with kb_controller.pressed(keyboard.Key.cmd):
            kb_controller.tap('v')
        
        # Give macOS a tiny fraction of a second to register the paste
        time.sleep(0.05)


def transcription_worker():
    global is_recording
    
    audio_buffer = []
    typed_char_count = 0  # We now track characters, not words
    last_process_time = time.time()
    
    while True:
        try:
            frame = audio_queue.get(timeout=0.05)
            audio_buffer.append(frame)
        except queue.Empty:
            pass

        if not is_recording:
            audio_buffer = []
            typed_char_count = 0
            continue

        current_time = time.time()
        
        if current_time - last_process_time >= UPDATE_INTERVAL and len(audio_buffer) > 0:
            last_process_time = current_time
            
            audio_data = np.concatenate(audio_buffer, axis=0).flatten()
            
            if len(audio_data) < SAMPLE_RATE * 0.5:
                continue

            # Notice we removed the initial_prompt and punctuation stripper
            result = mlx_whisper.transcribe(
                audio_data, 
                path_or_hf_repo=MODEL_REPO, 
                fp16=True, 
                language="en"
            )
            
            current_text = result.get("text", "").strip()
            
            # Block hallucinations
            lower_text = current_text.lower()
            hallucinations = ["thank you", "thanks for watching", "subtitles by"]
            if any(h in lower_text for h in hallucinations) and len(current_text.split()) < 4:
                continue
            
            # If the text has updated, highlight the old text and paste the new text
            if current_text:
                replace_text(typed_char_count, current_text)
                
                # Update our character count (plus 1 for the trailing space we add)
                typed_char_count = len(current_text) + 1

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

import sounddevice as sd
import numpy as np
import wave
import time
import os
import threading
from scipy.io import wavfile
import queue

# Constants
SAMPLE_RATE = 16000
CHANNELS = 1
SILENCE_THRESHOLD = 0.01  # Energy threshold for detecting speech
SILENCE_DURATION = 2.0     # Finalize a chunk after 2s of silence
CHUNK_FOLDER = "audio_chunks"

if not os.path.exists(CHUNK_FOLDER):
    os.makedirs(CHUNK_FOLDER)

class BackgroundRecorder:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.recording = False
        self.current_chunk = []
        self.silence_timer = 0
        self.chunk_id = 0

    def audio_callback(self, indata, frames, time_info, status):
        # Calculate energy (RMS)
        energy = np.sqrt(np.mean(indata**2))
        
        if energy > SILENCE_THRESHOLD:
            # We have sound
            self.current_chunk.extend(indata.flatten())
            self.silence_timer = 0
            if not self.recording:
                print("Recording started...")
                self.recording = True
        else:
            # Silence
            if self.recording:
                self.current_chunk.extend(indata.flatten())
                self.silence_timer += frames / SAMPLE_RATE
                
                if self.silence_timer >= SILENCE_DURATION:
                    # Finalize chunk
                    self.save_chunk()
                    self.recording = False

    def save_chunk(self):
        if len(self.current_chunk) == 0:
            return
            
        print(f"Finalizing chunk {self.chunk_id}...")
        data = np.array(self.current_chunk)
        filename = os.path.join(CHUNK_FOLDER, f"chunk_{int(time.time())}_{self.chunk_id}.wav")
        # Scale to 16-bit PCM
        scaled = np.int16(data / np.max(np.abs(data)) * 32767)
        wavfile.write(filename, SAMPLE_RATE, scaled)
        
        self.audio_queue.put(filename)
        self.current_chunk = []
        self.chunk_id += 1

    def listen(self):
        print(f"Listening with threshold {SILENCE_THRESHOLD} (24/7 Mode)...")
        with sd.InputStream(callback=self.audio_callback,
                           channels=CHANNELS,
                           samplerate=SAMPLE_RATE):
            while True:
                sd.sleep(1000)

if __name__ == "__main__":
    recorder = BackgroundRecorder()
    recorder.listen()

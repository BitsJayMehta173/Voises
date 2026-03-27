import subprocess
import time
import os

from dataset_cutter import cut_dataset

def start_system():
    # Ensure folders exist
    if not os.path.exists("audio_chunks"):
        os.makedirs("audio_chunks")
    if not os.path.exists("dataset"):
        os.makedirs("dataset")

    print("--- Starting Voice Recorder (Recording) ---")
    recorder = subprocess.Popen(["python", "audio_recorder.py"])
    
    print("--- Starting STT Processor (Transcription) ---")
    processor = subprocess.Popen(["python", "stt_processor.py"])
    
    # [NEW] Store PIDs for the toggle switch
    import json
    with open("pids.json", "w") as f:
        json.dump({"recorder": recorder.pid, "processor": processor.pid, "main": os.getpid()}, f)
    print("Speak to your microphone to start recording.")
    print("Every 5 minutes, the dataset will be automatically updated with new word segments.")
    print("Press Ctrl+C in this terminal to stop (Both processes).")

    last_cut_time = time.time()
    try:
        while True:
            # Check if any process has died
            if recorder.poll() is not None:
                print("Recorder stopped unexpectedly! Restarting...")
                recorder = subprocess.Popen(["python", "audio_recorder.py"])
            if processor.poll() is not None:
                print("Processor stopped unexpectedly! Restarting...")
                processor = subprocess.Popen(["python", "stt_processor.py"])
            
            # Periodically run the dataset cutter (every 5 minutes)
            if time.time() - last_cut_time > 300:
                print("\n[Auto-Task] Refreshing dataset from transcripts...")
                # We call the function directly if imported, or run it as a subprocess
                subprocess.run(["python", "dataset_cutter.py"])
                last_cut_time = time.time()
            
            time.sleep(30)
    except KeyboardInterrupt:
        print("\nStopping all processes...")
        recorder.terminate()
        processor.terminate()
        print("Done.")

if __name__ == "__main__":
    start_system()

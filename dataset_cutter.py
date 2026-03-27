import os
import sqlite3
import json
import pandas as pd
import wave
import numpy as np
import noisereduce as nr
from anyascii import anyascii

# Settings
DATABASE = "transcripts.db"
DATASET_PATH = "dataset"
AUDIO_OUT = os.path.join(DATASET_PATH, "audio")
METADATA = os.path.join(DATASET_PATH, "metadata.csv")
PADDING_MS = 60  
MAX_SAMPLES_PER_WORD = 50 

if not os.path.exists(AUDIO_OUT):
    os.makedirs(AUDIO_OUT)

def cut_wav(filepath, start_ms, end_ms, out_path):
    """Pure-python WAV cutting (No ffmpeg needed)"""
    with wave.open(filepath, 'rb') as w:
        params = w.getparams()
        # Set positions in frames
        start_frame = int(start_ms * params.framerate / 1000)
        end_frame = int(end_ms * params.framerate / 1000)
        
        w.setpos(start_frame)
        frames = w.readframes(end_frame - start_frame)
        
        # Audio Processing with Numpy
        samples = np.frombuffer(frames, dtype=np.int16)
        
        # 1. Light Noise Reduction
        try:
            samples = nr.reduce_noise(y=samples, sr=params.framerate, stationary=False)
        except:
            pass
            
        # 2. Peak Normalization (Custom implementation)
        peak = np.max(np.abs(samples))
        if peak > 0:
            samples = (samples / peak * 32767 * 0.9).astype(np.int16)
            
        with wave.open(out_path, 'wb') as out:
            out.setparams(params)
            out.writeframes(samples.tobytes())

def cut_dataset():
    conn = sqlite3.connect(DATABASE)
    df = pd.read_sql_query("SELECT * FROM transcriptions", conn)
    df = df.sort_values(by=["timestamp"], ascending=False)
    
    word_counts = {}
    entries = []
    
    for _, row in df.iterrows():
        chunk_file = row["chunk_file"]
        if not os.path.exists(chunk_file):
            continue
            
        try:
            with wave.open(chunk_file, 'rb') as w:
                audio_len_ms = (w.getnframes() / w.getparams().framerate) * 1000
                
            word_data = json.loads(row["word_data"])
            for i, item in enumerate(word_data):
                word = item["word"].strip().lower()
                prob = item["probability"]
                
                if len(word) < 2:
                    continue
                
                if word not in word_counts:
                    word_counts[word] = 0
                if word_counts[word] >= MAX_SAMPLES_PER_WORD:
                    continue
                
                start_ms = max(0, int(item["start"] * 1000) - PADDING_MS)
                end_ms = min(audio_len_ms, int(item["end"] * 1000) + PADDING_MS)
                
                # Cut, Normalize, Clean & Save
                base_name = os.path.basename(chunk_file).replace(".wav", "")
                out_name = f"{base_name}_w{i}_{word}.wav"
                out_path = os.path.join(AUDIO_OUT, out_name)
                
                cut_wav(chunk_file, start_ms, end_ms, out_path)
                
                # Metadata status
                is_consensus = (word in str(row["consensus_trans"]).lower())
                is_llm = (word in str(row["llm_verified"]).lower())
                status = "verified" if (is_consensus and is_llm and prob > 0.75) else "weak"

                word_counts[word] += 1
                entries.append({
                    "path": out_name,
                    "text": word,
                    "status": status,
                    "prob": prob,
                    "google_text": anyascii(str(row["google_trans"])).lower(),
                    "whisper_text": anyascii(str(row["whisper_trans"])).lower(),
                    "llm_text": anyascii(str(row["llm_verified"])).lower(),
                    "consensus_text": anyascii(str(row["consensus_trans"])).lower()
                })
                
        except Exception as e:
            print(f"Error processing {chunk_file}: {e}")
            
    final_df = pd.DataFrame(entries)
    final_df.to_csv(METADATA, index=False)
    print(f"Dataset Refined: {len(entries)} Exported. Independent of ffmpeg.")

if __name__ == "__main__":
    cut_dataset()

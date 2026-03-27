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
    try:
        with wave.open(filepath, 'rb') as w:
            params = w.getparams()
            # Set positions in frames
            start_frame = int(start_ms * params.framerate / 1000)
            end_frame = int(end_ms * params.framerate / 1000)
            
            w.setpos(start_frame)
            frames = w.readframes(max(0, end_frame - start_frame))
            if not frames:
                return False
                
            # Audio Processing with Numpy
            samples = np.frombuffer(frames, dtype=np.int16)
            
            # Simple Cleaning 
            try:
                samples = nr.reduce_noise(y=samples, sr=params.framerate, stationary=False)
            except:
                pass
                
            # Peak Normalize 
            peak = np.max(np.abs(samples)) if len(samples) > 0 else 0
            if peak > 0:
                samples = (samples / peak * 32767 * 0.9).astype(np.int16)
                
            with wave.open(out_path, 'wb') as out:
                out.setparams(params)
                out.writeframes(samples.tobytes())
            return True
    except Exception as e:
        print(f"Error in cut_wav: {e}")
        return False

def cut_dataset():
    if not os.path.exists(DATABASE):
        print("Database not found!")
        return
        
    conn = sqlite3.connect(DATABASE)
    df = pd.read_sql_query("SELECT * FROM transcriptions", conn)
    df = df.sort_values(by=["timestamp"], ascending=False)
    
    word_counts = {}
    entries = []
    
    print(f"Scanning {len(df)} database records...")
    
    for _, row in df.iterrows():
        chunk_file = row["chunk_file"]
        if not os.path.exists(chunk_file):
            continue
            
        try:
            with wave.open(chunk_file, 'rb') as w:
                audio_len_ms = (w.getnframes() / w.getparams().framerate) * 1000
                
            word_data = json.loads(row["word_data"])
            for i, item in enumerate(word_data):
                # Backwards compatibility check
                word_roman = item.get("word_roman", item.get("word", "")).strip().lower()
                word_native = item.get("word_native", "")
                prob = item.get("probability", 0)
                
                if len(word_roman) < 2:
                    continue
                
                if word_roman not in word_counts:
                    word_counts[word_roman] = 0
                if word_counts[word_roman] >= MAX_SAMPLES_PER_WORD:
                    continue
                
                start_ms = max(0, int(item["start"] * 1000) - PADDING_MS)
                end_ms = min(audio_len_ms, int(item["end"] * 1000) + PADDING_MS)
                
                # Cut, Normalize, Clean & Save
                base_name = os.path.basename(chunk_file).replace(".wav", "")
                out_name = f"{base_name}_w{i}_{word_roman}.wav"
                out_path = os.path.join(AUDIO_OUT, out_name)
                
                success = cut_wav(chunk_file, start_ms, end_ms, out_path)
                
                if success:
                    # Metadata status logic
                    # Check consensus or whisper_roman/whisper_native columns
                    is_consensus = (word_roman in str(row.get("consensus_trans", "")).lower())
                    is_llm = (word_roman in str(row.get("llm_verified_roman", "")).lower())
                    status = "verified" if (is_consensus and is_llm and prob > 0.75) else "weak"

                    word_counts[word_roman] += 1
                    entries.append({
                        "path": out_name,
                        "text_roman": word_roman,
                        "text_native": word_native,
                        "status": status,
                        "prob": prob,
                        "google_text": anyascii(str(row.get("google_trans", ""))).lower(),
                        "whisper_roman": anyascii(str(row.get("whisper_roman", ""))).lower(),
                        "whisper_native": str(row.get("whisper_native", "")),
                        "llm_text": anyascii(str(row.get("llm_verified_roman", ""))).lower()
                    })
                
        except Exception as e:
            print(f"Row {row['id']} processing error: {e}")
            
    if entries:
        final_df = pd.DataFrame(entries)
        final_df.to_csv(METADATA, index=False)
        print(f"SUCCESS: {len(entries)} verified/weak segments exported to {METADATA}")
    else:
        print("ALERT: No segments found to export! (Maybe no speech detected or threshold too high?)")

if __name__ == "__main__":
    cut_dataset()

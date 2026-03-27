import os
import sqlite3
import json
import pandas as pd
from pydub import AudioSegment, effects
import noisereduce as nr
import numpy as np

# Settings
DATABASE = "transcripts.db"
DATASET_PATH = "dataset"
AUDIO_OUT = os.path.join(DATASET_PATH, "audio")
METADATA = os.path.join(DATASET_PATH, "metadata.csv")
PADDING_MS = 60  
MAX_SAMPLES_PER_WORD = 50 # Avoid redundancy: max 50 samples for "the", "hai", etc.

if not os.path.exists(AUDIO_OUT):
    os.makedirs(AUDIO_OUT)

def cut_dataset():
    conn = sqlite3.connect(DATABASE)
    df = pd.read_sql_query("SELECT * FROM transcriptions", conn)
    
    # Sort by consensus and probability to keep the best samples if we hit the cap
    df = df.sort_values(by=["timestamp"], ascending=False)
    
    word_counts = {}
    entries = []
    
    for _, row in df.iterrows():
        chunk_file = row["chunk_file"]
        if not os.path.exists(chunk_file):
            continue
            
        try:
            audio = AudioSegment.from_wav(chunk_file)
            audio_len = len(audio)
            word_data = json.loads(row["word_data"])
            
            for i, item in enumerate(word_data):
                word = item["word"].strip().lower()
                prob = item["probability"]
                
                # Filter: Word length and confidence
                if prob < 0.65 or len(word) < 2:
                    continue
                
                # Check Diversity Cap
                if word not in word_counts:
                    word_counts[word] = 0
                if word_counts[word] >= MAX_SAMPLES_PER_WORD:
                    continue
                
                # Timing with padding
                start_ms = max(0, int(item["start"] * 1000) - PADDING_MS)
                end_ms = min(audio_len, int(item["end"] * 1000) + PADDING_MS)
                
                # 1. Extraction
                segment = audio[start_ms:end_ms]
                
                # 2. Loudness Normalization (Peak normalize to -1.0 dB)
                segment = effects.normalize(segment)
                
                # 3. Noise Reduction (Light cleaning)
                # Convert to numpy for noisereduce
                samples = np.array(segment.get_array_of_samples())
                # Use a non-stationary noise reduction (good for fan noise/laptop hum)
                cleaned_samples = nr.reduce_noise(y=samples, sr=segment.frame_rate, stationary=False)
                # Convert back to audio segment
                segment = segment._spawn(cleaned_samples.astype(np.int16).tobytes())

                # 4. Save
                base_name = os.path.basename(chunk_file).replace(".wav", "")
                out_name = f"{base_name}_w{i}_{word}.wav"
                out_path = os.path.join(AUDIO_OUT, out_name)
                
                segment.export(out_path, format="wav")
                
                word_counts[word] += 1
                entries.append({
                    "path": out_name,
                    "text": word,
                    "prob": prob,
                    "llm_verified": word in row["llm_verified"].lower(),
                    "consensus_verified": word in row["consensus_trans"].lower()
                })
                
        except Exception as e:
            print(f"Error processing {chunk_file}: {e}")
            
    # Save metadata
    final_df = pd.DataFrame(entries)
    final_df.to_csv(METADATA, index=False)
    print(f"Dataset Refined: {len(entries)} cleaned segments saved. Diversity cap applied.")

if __name__ == "__main__":
    cut_dataset()

if __name__ == "__main__":
    cut_dataset()

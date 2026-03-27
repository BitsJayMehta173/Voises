import os
import time
import queue
import threading
import sqlite3
import speech_recognition as sr
from faster_whisper import WhisperModel
import json
import vosk
import wave
from anyascii import anyascii
from transformers import pipeline
from pydub import AudioSegment, effects
import shutil

# Settings
CHUNK_FOLDER = "audio_chunks"
DATASET_FOLDER = "dataset"
TTS_DATA_FOLDER = os.path.join(DATASET_FOLDER, "tts_data")
TTS_METADATA = os.path.join(DATASET_FOLDER, "tts_metadata.csv")
DATABASE = "transcripts.db"
WHISPER_MODEL_SIZE = "base"

if not os.path.exists(TTS_DATA_FOLDER):
    os.makedirs(TTS_DATA_FOLDER)

# DB Setup
conn = sqlite3.connect(DATABASE, check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS transcriptions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chunk_file TEXT,
        google_trans TEXT,
        whisper_roman TEXT,
        whisper_native TEXT,
        consensus_trans TEXT,
        llm_verified_roman TEXT,
        llm_verified_native TEXT,
        word_data TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
""")
conn.commit()

class STTProcessor:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        print(f"Loading Whisper model ({WHISPER_MODEL_SIZE})...")
        self.whisper = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
        
        # LLM based verification 
        print("Loading Contextual Brain (BERT)...")
        self.mlm = pipeline("fill-mask", model="bert-base-multilingual-cased", top_k=1)

    def verify_with_llm(self, text):
        if not text or len(text.split()) < 3:
            return text
        words = text.split()
        verified_words = []
        for i in range(len(words)):
            original_word = words[i]
            masked_sentence = " ".join(words[:i] + ["[MASK]"] + words[i+1:])
            try:
                result = self.mlm(masked_sentence)
                top_prediction = result[0]['token_str'].lower()
                if original_word.lower() in top_prediction or top_prediction in original_word.lower():
                    verified_words.append(original_word)
                else:
                    verified_words.append(original_word)
            except:
                verified_words.append(original_word)
        return " ".join(verified_words)

    def process_chunk(self, filepath):
        print(f"Processing {filepath}...")
        results = {"chunk_file": filepath}
        
        # 1. Google Web STT (with en-IN support)
        try:
            with sr.AudioFile(filepath) as source:
                audio = self.recognizer.record(source)
                try:
                    results["google"] = self.recognizer.recognize_google(audio, language="en-IN")
                except:
                    results["google"] = self.recognizer.recognize_google(audio)
        except Exception as e:
            results["google"] = ""

        # 2. Faster-Whisper (ChatGPT engine)
        try:
            segments, info = self.whisper.transcribe(filepath, word_timestamps=True, task="transcribe")
            word_list = []
            full_text_roman = []
            full_text_native = []
            for segment in segments:
                native_text = segment.text.strip()
                roman_text = anyascii(native_text)
                full_text_roman.append(roman_text)
                full_text_native.append(native_text)
                for word in segment.words:
                    word_list.append({
                        "word_roman": anyascii(word.word),
                        "word_native": word.word,
                        "start": word.start,
                        "end": word.end,
                        "probability": word.probability
                    })
            results["whisper_roman"] = " ".join(full_text_roman)
            results["whisper_native"] = " ".join(full_text_native)
            results["word_data"] = json.dumps(word_list)
        except Exception as e:
            results["whisper_roman"] = ""
            results["whisper_native"] = ""
            results["word_data"] = "[]"

        # 3. LLM Verifier
        try:
            results["llm_verified_roman"] = self.verify_with_llm(results["whisper_roman"])
            results["llm_verified_native"] = results["whisper_native"] # Keep native as is
        except:
            results["llm_verified_roman"] = results["whisper_roman"]
            results["llm_verified_native"] = results["whisper_native"]

        # Consensus logic
        verified_text = []
        w_text = results["whisper_roman"].lower()
        g_text = results["google"].lower()
        for w_json in json.loads(results["word_data"]):
            word = w_json["word_roman"].strip().lower()
            if word in g_text or word in w_text:
                verified_text.append(word)
        results["consensus"] = " ".join(verified_text)

        # 🚀 [NEW] TTS DATA SAVE (Full Sentence/Sentence Level)
        # We only save to TTS if the sentence is clear (whisper or google found something)
        final_sentence = results["llm_verified_roman"].strip()
        if len(final_sentence.split()) >= 3:
            try:
                # 1. Clean and Normalize Audio for TTS (Pure Python)
                with wave.open(filepath, 'rb') as w:
                    params = w.getparams()
                    frames = w.readframes(w.getnframes())
                    samples = np.frombuffer(frames, dtype=np.int16)
                    
                    # Peak Normalize 
                    peak = np.max(np.abs(samples))
                    if peak > 0:
                        samples = (samples / peak * 32767 * 0.9).astype(np.int16)
                    
                    tts_filename = os.path.basename(filepath)
                    tts_path = os.path.join(TTS_DATA_FOLDER, tts_filename)
                    with wave.open(tts_path, 'wb') as tts_out:
                        tts_out.setparams(params)
                        tts_out.writeframes(samples.tobytes())
                
                # 2. Update TTS metadata
                header = "path,text\n"
                if not os.path.exists(TTS_METADATA):
                    with open(TTS_METADATA, "w", encoding="utf-8") as f:
                        f.write(header)
                
                with open(TTS_METADATA, "a", encoding="utf-8") as f:
                    f.write(f"{tts_filename},{final_sentence}\n")
                print(f"TTS Sample Saved: {final_sentence[:30]}...")
            except Exception as e:
                print(f"Error saving TTS sample: {e}")

        # Save to DB
        cursor.execute("""
            INSERT INTO transcriptions (chunk_file, google_trans, whisper_roman, whisper_native, consensus_trans, llm_verified_roman, llm_verified_native, word_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (filepath, results["google"], results["whisper_roman"], results["whisper_native"], results["consensus"], results["llm_verified_roman"], results["llm_verified_native"], results["word_data"]))
        conn.commit()
        print(f"STT Verified: {results['llm_verified_roman'][:30]}...")

    def run(self):
        print("Dual STT/TTS System monitoring folder...")
        processed = set()
        while True:
            files = [f for f in os.listdir(CHUNK_FOLDER) if f.endswith(".wav")]
            for f in files:
                path = os.path.join(CHUNK_FOLDER, f)
                if path not in processed:
                    self.process_chunk(path)
                    processed.add(path)
            time.sleep(5)

if __name__ == "__main__":
    processor = STTProcessor()
    processor.run()

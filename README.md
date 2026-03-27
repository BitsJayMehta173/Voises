# 🎤 24/7 Expert Voice Dataset Collector (Pro-Edition)

This is a state-of-the-art, automated background system designed to build high-quality datasets for **Speech Recognition (STT)** and **Speech Synthesis/Voice Cloning (TTS)**. Specifically engineered for **Indian Accents**, **Hinglish**, and **Devnagri Script Support**.

## 🛠️ The Expert Pipeline Architecture

1.  **`audio_recorder.py` (Smart Listener)**:
    - Runs in the background with **Voice Activity Detection (VAD)**.
    - Captures high-quality 24-bit 44.1kHz or 16-bit 16kHz speech chunks.
    - **Zero-Data Loss**: Only saves when you are actively speaking.

2.  **`stt_processor.py` (The Multi-Engine Brain)**:
    - **AI Committee**: Transcribes every spoken phrase using **Google STT (en-IN)** and **OpenAI Whisper (Local)**.
    - **Multi-Script Storage**: Saves both **Romanized English** (e.g., `mera naam`) and **Native Devnagri Script** (e.g., `मेरा नाम`).
    - **LLM Verification**: Uses a **BERT Multilingual Brain** to check if your speech "makes sense" in context.
    - **TTS Dataset**: Automatically saves full, normalized sentences for voice cloning into `dataset/tts_data/`.

3.  **`dataset_cutter.py` (The Audio Lab - Zero Dependency)**:
    - Extracts individual words from speech with **60ms of padding**.
    - **Pure-Python Normalization**: No external FFmpeg required.
    - **AI Noise Reduction** (`noisereduce`): Automatically removes laptop fans, typing sounds, and background hum.
    - **Multi-Model Metadata**: Keeps a record of every engine's "opinion" for every word clip.

4.  **`toggle_system.py` (The Efficiency Switch)**:
    - A "Pause/Play" toggle that **suspends** background processes.
    - **Saves 100% CPU** while paused, but keeps the heavy AI models in memory for instant resumption.

5.  **`dashboard.py` (Human-in-the-Loop Reviewer)**:
    - **Expert Mode Review**: Shows Google Thought, Whisper Thought, and BERT Thought side-by-side.
    - **Manual Correction**: Edit any transcript and save it as a "Corrected" (Gold Standard) sample.
    - **Status Filtering**: Filter by "Weak," "Verified," or "Corrected" to focus your efforts.

---

## 🚀 How to Use

1.  **Initialize & Start Recording**:
    ```bash
    python run_all.py
    ```

2.  **Pause/Resume Listening (CPU Saver)**:
    ```bash
    python toggle_system.py
    ```

3.  **Expert Review & Manual Correction**:
    - Build your dataset first, then run:
    ```bash
    python dashboard.py
    ```
    - Visit **`http://127.0.0.1:7860/`** to verify and edit your data.

---

## 📂 Dataset Locations
- **STT (Word clips)**: `dataset/audio/` + `metadata.csv` (Roman and Native script included).
- **TTS (Sentences)**: `dataset/tts_data/` + `tts_metadata.csv` (High-fidelity full recordings).

## 🔒 Privacy & Speed
All heavy transcription (Whisper/BERT) happens **locally on your machine**. Your voice recordings never leave your laptop unless you explicitly share them.

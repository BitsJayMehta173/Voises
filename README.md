# 🎤 24/7 Voice Dataset Collector (Pro-Edition)

This is a comprehensive, automated background system designed to build high-quality datasets for both **Speech Recognition (STT)** and **Speech Synthesis/Voice Cloning (TTS)**, specifically optimized for **Indian Accents** and **Hinglish**.

## 🛠️ The Pipeline Architecture

1.  **`audio_recorder.py` (The Listener)**:
    - Runs in the background with **Voice Activity Detection (VAD)**.
    - Captures speech "chunks" (5-10s) only when you are talking.
    - Ignores silence to save disk space.

2.  **`stt_processor.py` (The Multi-Engine Brain)**:
    - **AI Consensus**: Transcribes each chunk using **Google STT (en-IN)** and **OpenAI Whisper (Local)**.
    - **Romanization**: Automatically converts any Hindi/Urdu script (e.g., `नाम`) into English letters (`naam`) using `anyascii`.
    - **Context Verification (LLM)**: Uses a **BERT Masked Language Model** to verify if words "make sense" in the context of the sentence.
    - **TTS Dataset**: Automatically saves full, normalized sentences for voice synthesis into `dataset/tts_data/`.

3.  **`dataset_cutter.py` (The Refiner)**:
    - Extracts individual words from speech with **60ms of padding**.
    - Applies **Loudness Normalization** (-1.0dB) to all clips.
    - Uses **AI Noise Reduction** (`noisereduce`) to remove laptop fans and keyboard sounds.
    - **Diversity Capping**: Limits each word to 50 high-quality samples to prevent dataset bias.

4.  **`toggle_system.py` (The Switch)**:
    - A "Pause/Play" toggle that **suspends** the background processes. It saves 100% of CPU during breaks but keeps the AI models in memory for instant resumption.

5.  **`dashboard.py` (The Reviewer)**:
    - A **Gradio-based web dashboard** to listen to, verify, or delete your collected samples.

---

## 🚀 How to Use

1.  **Initialize & Start**:
    ```bash
    python run_all.py
    ```

2.  **Pause/Resume Listening**:
    ```bash
    python toggle_system.py
    ```

3.  **Review Your Data**:
    ```bash
    python dashboard.py
    ```

4.  **Your Final Dataset**:
    - **STT (Words)**: `dataset/audio/` + `metadata.csv`
    - **TTS (Sentences)**: `dataset/tts_data/` + `tts_metadata.csv`

---

## 🔒 Privacy & Optimization
- **Privacy**: High-level transcription (Whisper and BERT) happens **locally on your machine**. 
- **Feasibility**: Designed for 3-4 hours of daily use. 30 hours of data is the "Sweet Spot" for a perfect personal voice model.

# AI-Powered-Multilingual-Voice-Interview-Simulator


# AI Interview Coach — Local Setup & Usage

## Table of Contents
- Overview
- Features
- Requirements
  - 1) Python & OS
  - 2) Python Packages
  - 3) External Tools
  - 4) Environment Variables / PATH
  - 5) API Keys
- Project Files Expected
- Optional: MFA Alignment for Pause Metrics
- How to Run
- Troubleshooting
- Security Note
- Sample requirements.txt

---

## Overview
An interactive, voice-based interview trainer that:
- asks context-aware questions (based on job title, job description, and resume),
- records spoken answers (Malayalam / English / Kannada),
- cleans/enhances audio (noise reduction, dereverb, band-pass, loudness),
- transcribes with OpenAI and optionally translates to English,
- analyzes speech (WPM, fillers, pauses via MFA),
- optionally compares an answer to a model answer for feedback.

---

## Features
- **Languages:** en, ml, kn (Whisper language detection + GPT verification)
- **Audio pipeline:** noisereduce → **WPE dereverb** (nara-wpe) → band-pass (≈300–3400 Hz) → **loudness normalization** → *(optional)* DL enhancement (Torch model)
- **Transcription:** gpt-4o-transcribe (primary) with gpt-4o-mini-transcribe (fallback for language check)
- **TTS (optional):** pyttsx3 to read questions / model answers
- **Resume parsing:** PyMuPDF (fitz)
- **Pause/Filler analysis:** custom filler lists for en/ml/kn + **(optional)** Montreal Forced Aligner (MFA) JSON

---

## Requirements

### 1) Python & OS
- **Python:** 3.9–3.11 recommended  
- **OS:** Windows / macOS / Linux  
- **Microphone access:** Required (sounddevice / PortAudio)  
- **GPU (optional):** for faster Whisper/Torch; install CUDA-enabled PyTorch if available.

### 2) Python Packages
Create and activate a virtual environment, then install:
    pip install --upgrade pip
    pip install openai-whisper openai sounddevice numpy scipy pyttsx3 librosa pydub PyMuPDF deep-translator soundfile noisereduce pyloudnorm nara-wpe torch torchaudio hyperpyyaml ipython

Notes:
- openai-whisper is used for language detection and basic ASR utilities.
- torch / torchaudio wheels should match OS/Python (and CUDA if using a GPU).
- pyttsx3 uses system TTS (SAPI5 on Windows, NSSpeechSynthesizer on macOS, eSpeak on Linux).

### 3) External Tools
- **FFmpeg** — required by pydub/librosa. Install and ensure the ffmpeg/bin directory is on PATH.
- **Montreal Forced Aligner (MFA)** *(optional; for pause/phone-level metrics)* — install MFA + an English acoustic model & dictionary. Ensure the MFA executable folder is on PATH.

### 4) Environment Variables / PATH
Windows (PowerShell) examples (adjust paths as needed):
    # Matplotlib backend for headless usage
    setx MPLBACKEND "Agg"

    # Montreal Forced Aligner (optional)
    setx MFA_ROOT_DIR "C:\Tools\MFA"
    setx PATH "C:\Tools\MFA\Library\bin;%PATH%"

    # FFmpeg
    setx PATH "C:\Tools\ffmpeg\bin;%PATH%"

Optional – DL speech enhancement model directory:
    C:/models/speech_enhance
This folder should contain the model/config expected by the enhancement loader (path is configurable in the code).

### 5) API Keys
- **OpenAI** — required for transcription, Q&A, and feedback.

macOS / Linux:
    export OPENAI_API_KEY="sk-..."

Windows (PowerShell):
    setx OPENAI_API_KEY "sk-..."

Models used in code (configure as needed):
- gpt-4o (question generation, answer comparison, language verification)
- gpt-4o-transcribe (main transcription)
- gpt-4o-mini-transcribe (fallback language check)

---

## Project Files Expected
- **Main notebook/script:** og-Copy4.ipynb (or an exported script).  
  To export and run as a script:
      jupyter nbconvert --to script og-Copy4.ipynb
      python og-Copy4.py

- **History files:**  history.json (auto-created in the working directory)
- **Temp audio:** temp_cleaned.wav, voice_after_cleaning.wav, and recordings like answer_basic_try1.wav (auto-created)

---

## Optional: MFA Alignment for Pause Metrics
To compute accurate pause counts/durations and word timings:
1. Align answer audio with its transcript using MFA.  
2. Export alignment to JSON (or convert TextGrid → JSON).  
3. Configure the code to read this JSON (the parser expects tiers.phones.entries and tiers.words.entries).

---

## How to Run

1) Start the program
    python og-Copy4.py
   (or run the notebook cell containing the entry point)

2) Provide inputs (CLI prompts)
- Job Title (required)
- Job Description (optional)
- Upload Resume? (yes/no → provide PDF path; text parsed via PyMuPDF)
- Experience Level: Fresher / Fresher with Internship / Work Experience
- Interview Type: choose a predefined type (Technical/Behavioral/etc.) or enter a custom type
- Mode: live (one-shot Q&A) or recorded (answer-by-answer with re-record option)
- Language: ml, en, or kn

3) Recording
- Choose timed recording (enter seconds) or manual stop (press ENTER to stop).
- Listen back and re-record until satisfied; then confirm to keep the attempt.

4) Automatic pipeline
- Audio cleanup: noise reduction → dereverb (WPE) → band-pass → loudness normalization → (optional) DL enhancement
- Language detection: Whisper (+ optional GPT verification)
- Transcription: OpenAI (preserves fillers, hesitations, pauses)
- Translation: if Malayalam/Kannada, translate to English via deep-translator
- Model answer (optional): generate a reference answer
- Answer comparison (optional): structured feedback vs. model answer
- Voice analytics: WPM, filler ratio/count, hesitations; (optional) pauses from MFA JSON

5) Outputs
- Console feedback + saved JSON history (questions, answers, metrics)
- Temp WAV files for raw and cleaned audio

---

## Troubleshooting
- Microphone not found / permission errors → check OS mic permissions; ensure sounddevice lists an input device.
- FFmpeg not found → install FFmpeg and add its bin folder to PATH.
- pyttsx3 silent → on Linux, install espeak/espeak-ng; macOS uses NSSpeech; Windows uses SAPI5.
- GPU not used → install a CUDA-enabled torch/torchaudio that matches the system GPU & drivers.
- MFA JSON missing → run MFA alignment and point the analysis step to the generated JSON (phones + words tiers).

---

## Security Note
Avoid hardcoding API keys. Load from environment variables or a .env file and exclude secrets from version control.

---

## Sample requirements.txt
    openai-whisper
    openai>=1.30.0
    sounddevice
    numpy
    scipy
    pyttsx3
    librosa
    pydub
    PyMuPDF
    deep-translator
    soundfile
    noisereduce
    pyloudnorm

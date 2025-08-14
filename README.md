# 🎙️ AI Audio Interview Coach

An end-to-end **audio interview training** tool that:
- Generates **context-aware questions** (uses your Job Title, Job Description, Résumé, Experience Level).
- Lets you **answer by voice** (Malayalam / English / Kannada).
- **Cleans & enhances audio** (noise reduction → WPE dereverb → band-pass → LUFS normalization → *(optional)* deep speech enhancement).
- **Detects language** (Whisper) and **transcribes** with OpenAI (preserves filler words, hesitations, `...` pauses).
- **Translates** non-English answers to English for comparable feedback.
- **Analyzes delivery** (WPM, filler ratio, long pauses via MFA alignment) and provides **AI feedback** by comparing to a model answer.
- Supports **Live mode** (continuous Q&A) and **Recorded mode** (retry takes and keep the best).

---

## 📚 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Requirements](#-requirements)
  - [1) Python & OS](#1-python--os)
  - [2) Python Packages](#2-python-packages)
  - [3) External Tools](#3-external-tools)
  - [4) Environment Variables / PATH](#4-environment-variables--path)
  - [5) API Keys](#5-api-keys)
- [Download the Code](#-download-the-code)
- [Local Setup (Step-by-Step)](#-local-setup-step-by-step)
- [How to Run](#-how-to-run)
- [Config You May Need to Edit](#-config-you-may-need-to-edit)
- [Outputs](#-outputs)
- [Optional: MFA Alignment](#-optional-mfa-alignment)
- [Troubleshooting](#-troubleshooting)
- [Security Note](#-security-note)
- [Sample requirements.txt](#-sample-requirementstxt)


---

## 🔎 Overview
This project implements a **single-notebook pipeline** that simulates realistic interview sessions. It asks **increasingly challenging** questions tailored to your context, records answers from your **microphone**, runs a **robust audio preprocessing chain**, performs **language detection + exact transcription** (preserving fillers & pauses), optionally **translates** to English, and computes **speaking-delivery metrics** (WPM, filler ratio, pause structure). It then provides **structured AI feedback** by comparing your answer to a generated **reference answer**.

---

## ✨ Features
- **Languages:** `en`, `ml`, `kn`  
  (Whisper for language ID + GPT verification fallback)
- **Audio pipeline:** `noisereduce` → **WPE dereverb** (`nara_wpe`) → **band-pass** (≈300–3400 Hz) → **loudness normalization** (≈ −23 LUFS) → *(optional)* **DL enhancement** (Torch model via `hyperpyyaml`)
- **Transcription:** `gpt-4o-transcribe` (preserves fillers & pauses)
- **Translation:** `deep-translator` (Malayalam/Kannada → English)
- **TTS (optional):** `pyttsx3` can read questions/model answers aloud
- **Résumé parsing:** `PyMuPDF (fitz)` reads PDF and injects context
- **Analytics:** speaking rate (WPM), filler ratio/count, pause counts/durations (via **MFA JSON**)
- **Modes:** **Live** (continuous Q&A) and **Recorded** (re-record per question)
- **Persistence:** session history saved to `history.json`

---

## ✅ Requirements

### 1) Python & OS
- **Python:** 3.9 – 3.11 recommended  
- **OS:** Windows / macOS / Linux  
- **Microphone:** required (`sounddevice` / PortAudio)  
- **GPU (optional):** for Torch/Whisper speedups (install CUDA-matching wheels)

### 2) Python Packages
Create a virtual environment and install dependencies:

```bash
# Create and activate (choose one)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Upgrade pip
pip install -U pip

# PyTorch/torchaudio (CPU baseline; see https://pytorch.org for CUDA builds)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Core packages
pip install -U \
  openai-whisper openai sounddevice scipy numpy pyttsx3 librosa pydub \
  pymupdf deep-translator soundfile noisereduce pyloudnorm nara_wpe \
  torchaudio hyperpyyaml ipython
```

### Notes
 - `openai-whisper` requires FFmpeg on PATH.
 - `pyttsx3` uses OS TTS backends (Windows=SAPI5, macOS=NSSpeech, Linux=eSpeak).
 - `nara_wpe` provides dereverberation (WPE).
 - If using GPU, install `torch/torchaudio` per your CUDA version from the official site.


### 3) External Tools
- **FFmpeg** — required by Whisper/PyDub/Librosa
   - Windows: download FFmpeg and add `...\ffmpeg\bin` to PATH
   - macOS: `brew install ffmpeg`
   - Linux (Debian/Ubuntu): `sudo apt-get install -y ffmpeg`
- **PortAudio** — backend for sounddevice
   - macOS: `brew install portaudio`
   - Linux: `sudo apt-get install -y portaudio19-dev`
-  **Montréal Forced Aligner (MFA)** — for robust pause/phone timings
Install MFA and acoustic model(s); ensure mfa is on PATH


### 4) Environment Variables / PATH

**macOS / Linux (bash/zsh):**
```bash
# Recommended: headless plotting backend
export MPLBACKEND="Agg"

# OpenAI key (see section 5)
export OPENAI_API_KEY="sk-..."

# FFmpeg
export PATH="/usr/local/bin:$PATH"           # if brew installed ffmpeg
# or if you extracted ffmpeg to a custom dir, add its 'bin':
export PATH="$HOME/tools/ffmpeg/bin:$PATH"

# (Optional) MFA
export MFA_ROOT_DIR="$HOME/.local/share/mfa"
export PATH="$HOME/miniconda3/envs/mfa/bin:$PATH"   # adjust to your install
```

**Windows (PowerShell):**
```bash
# Headless plotting backend
setx MPLBACKEND "Agg"

# OpenAI key (see section 5)
setx OPENAI_API_KEY "sk-..."

# FFmpeg (example path)
setx PATH "C:\tools\ffmpeg\bin;%PATH%"

# (Optional) MFA (examples; adjust to your install)
setx MFA_ROOT_DIR "C:\Users\<you>\Documents\MFA"
setx PATH "C:\code_projects\MFA\Library\bin;%PATH%"
# If using a specific exe path in code, ensure it exists (see config section)

```
The notebook includes Windows-style PATH samples such as:

`C:\code_projects\MFA\Library\bin`
`C:\code_projects\ffmpeg_release_full\ffmpeg-7.1.1-full_build\bin`
Update these for your machine or prefer adding `mfa/ffmpeg` to PATH globally.

### 5) API Keys

**OpenAI** — required for transcription, Q&A and feedback.

**macOS / Linux:**
`export OPENAI_API_KEY="sk-..."`

**Windows (PowerShell):**
setx OPENAI_API_KEY "sk-..."

**In code, prefer:**
```bash
from openai import OpenAI
client = OpenAI()  # reads OPENAI_API_KEY from env
```

(The notebook originally uses `OpenAI(api_key="YOUR_GPT_API_KEY")`; you can replace that with the snippet above.)

---
### ⬇️ Download the Code

Clone this repo or download the ZIP:

```bash
# Using git
git clone https://github.com/<your-account>/<your-repo>.git
cd <your-repo>

# OR download ZIP from GitHub and extract it
```
Main file: A `udio_Interview.ipynb` (the end-to-end pipeline).


## 🛠 Local Setup (Step-by-Step)
1 - Install Python 3.9–3.11 and Git.

2 - Clone the repository (or extract ZIP).

3 - Create and activate a virtual environment (see Python Packages).

4 - Install required Python packages.

5 - Install FFmpeg and ensure ffmpeg is on PATH (ffmpeg -version should work).

6 - (Optional) Install MFA and a suitable acoustic model.

7 - Set OPENAI_API_KEY in your environment.

8 - (Windows only) If you will use the notebook’s hard-coded MFA path, ensure the exe exists (or change the code to call mfa directly).

9 - Run Jupyter or export the notebook to a script (see next section).

---
### ▶️ How to Run
**Option A — Run in Jupyter**
```bash
# From the repo folder
jupyter lab     # or: jupyter notebook
```
Open `Audio_Interview.ipynb`, run cells top-to-bottom, then execute the final entry cell to start the interactive prompts.

During a session you’ll be asked for:
 - Job Title (required)
 - Job Description (optional)
 - Upload Résumé? (yes/no → provide PDF path; text parsed via PyMuPDF)
 - Experience Level: Fresher / Fresher with Internship / Work Experience
 - Interview Type: choose a predefined type (Behavioral/Technical/etc.) or enter a custom one
 - Mode: live or recorded
 - Language: ml, en, or kn

**Option B — Export to Script and Run**
```bash
jupyter nbconvert --to script Audio_Interview.ipynb
python Audio_Interview_converted.py
```
Follow the same CLI prompts as in Jupyter.

---
### ⚙️ Config You May Need to Edit
 - OpenAI client: Prefer client = OpenAI() (reads env var) instead of hardcoding.
 - Whisper model for LID: default is "small"; you can switch to "base"/"medium" per speed/accuracy needs.
 - MFA executable path (Windows example): the notebook invokes a path like C:\code_projects\MFA\Scripts\mfa.exe.
 - If mfa is on PATH globally, change the code to call "mfa" instead of a hard-coded path.
 - Acoustic model identifiers: code selects by language (e.g., "english_mfa" and "tamil_cv").
 - Ensure those names match models installed in your MFA. Otherwise, update the identifiers in code.
 - Deep speech enhancement model directory: default example is:

```
C:/code_projects/RP2/pretrained_models/enhance
```
This directory should include `hyperparams.yaml` and `enhance_model.ckpt`.
Update preprocess_audio_pipeline(voice_file, model_dir="...") to your folder or temporarily disable enhancement by commenting out the enhancement step and using the cleaned intermediate file.

---
### 📤 Outputs
 - Temp audio: response.wav, temp_cleaned.wav, voice_after_cleaning.wav, and per-question recordings (e.g., answer_2_try1.wav)
 - Session history: history.json — contains asked questions and your answers (and can be extended with metrics)
 - Console/UI:
Reference model answer (optional)
Structured comparison feedback (optional)
Voice analytics (WPM, filler ratio, pause stats) (requires MFA for pauses)

---
### ⏱ Optional: MFA Alignment
```bash
# 1. Create and activate a Conda environment
conda create -n mfa_env python=3.10 -y
conda activate mfa_env

# 2. Install MFA from conda-forge
conda install -c conda-forge montreal-forced-aligner -y

# 3. Verify installation
mfa version

# 4. Download example pre-trained models
mfa model download acoustic english
mfa model download dictionary english_us_arpa
```

To compute accurate pause counts/durations and word timings:

 1 - Install MFA and download an appropriate acoustic model & dictionary (e.g., English; for Dravidian languages choose a close model or adjust).
 
 2 - Ensure the MFA executable is callable (either on PATH or via the hard-coded path you set).
 
 3 - The pipeline writes transcripts and calls MFA to produce JSON alignment.
 
 4 - The analyzer expects phone entries to compute pause durations (silences ≥ configurable threshold).
 
**If MFA isn’t installed, choose “no” when prompted for voice analysis.**

---
### 🧩 Troubleshooting

**FFmpeg not found**
```bash
OSError: [Errno 2] No such file or directory: 'ffmpeg'
```
Install FFmpeg and ensure its bin is on PATH.

**Microphone / PortAudio errors**
```bash
sounddevice.PortAudioError: Error opening InputStream
```
Check OS mic permissions and that an input device exists.
Install PortAudio (brew install portaudio or sudo apt-get install portaudio19-dev).

**TTS silent on Linux**
```bash
pyttsx3 did not speak
```
Install espeak / espeak-ng and try again.

**CUDA/Torch mismatch**
- Install the correct torch/torchaudio for your CUDA driver from https://pytorch.org.
- For CPU-only, use the --index-url command shown in the install section.

**MFA alignment fails / JSON missing**
- Ensure `mfa` is on PATH or update the path the notebook uses.
- Verify installed model names match what code references (e.g., `english_mfa`).
- If unneeded, skip MFA by selecting “no” for voice analysis.

**OpenAI authentication**
```bash
openai.AuthenticationError / No API key provided
```
Set `OPENAI_API_KEY` env var and ensure `client = OpenAI()` is used.

---
### 🔐 Security Note
Do not hardcode API keys. Prefer environment variables or a `.env` file (excluded from version control).
Remove temp WAV files and `history.json` after sessions if you do not want to keep local artifacts.

---
### 📄 Sample requirements.txt
```bash
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
nara-wpe
torch
torchaudio
hyperpyyaml
ipython
```


# audio_interview_coach.py
import re
import os
import json
import time
import queue
import threading
import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import pyttsx3
import librosa
import IPython.display as ipd
from pydub import AudioSegment
from datetime import datetime
from openai import OpenAI
import fitz  # PyMuPDF for CV extraction
from deep_translator import GoogleTranslator
import soundfile as sf
import noisereduce as nr
import pyloudnorm as pyln
import nara_wpe.wpe as wpe
from scipy.signal import butter, lfilter
import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml
import IPython.display as ipd
import shutil
import subprocess
import os, json, uuid, time
from typing import Dict, Any, List

# --------------------------- CONFIGURATION ---------------------------
client = OpenAI(api_key="GPT_API_KEY")  # Replace with your key
MODEL = whisper.load_model("small")  # Accurate + lightweight for LID
INTERVIEW_HISTORY_FILE = "interview_history.json"
SUPPORTED_LANGUAGES = ["en", "ml", "kn"]  # English, Malayalam, Kannada


# -----------------------------
# Environment Setup for MFA
# -----------------------------

os.environ["MPLBACKEND"] = "Agg"
os.environ["MFA_ROOT_DIR"] = r"C:\Users\moham\Documents\MFA"
os.environ["PATH"] = r"C:\code_projects\MFA\Library\bin;" + os.environ["PATH"]
os.environ["PATH"] = r"C:\code_projects\ffmpeg_release_full\ffmpeg-7.1.1-full_build\bin;" + os.environ["PATH"]

# --------------------------- FILLER WORDS ---------------------------
FULL_FILLER_WORDS_EN = [
    "um", "uh", "like", "you know", "so", "actually", "basically", "okay",
    "right", "well", "hmm", "ah", "oh", "just", "literally", "honestly", "really", "seriously"
]

FULL_FILLER_WORDS_ML = [  # Malayalam filler words
    "അല്ലേ", "പോലെ", "അതായത്", "എന്താണെന്ന്", "അതെ", "അല്ല", "ഇല്ലേ", "ശരി", "ഹം", "ആ", "പറഞ്ഞാൽ"
]

FULL_FILLER_WORDS_KN = [  # Kannada filler words
    "ಅಂತ", "ಅದೇ", "ಹೀಗಾಗಿ", "ಅದು", "ಹೌದು", "ಇಲ್ಲ", "ಅಯ್ಯೋ", "ಸರಿ", "ಓ", "ಹುಂ"
]


PREDEFINED_QUESTIONS = {
    "Behavioral Interview": [
        "Tell me about yourself.",
        "Can you describe a recent challenge you faced and how you handled it?"
    ],
    "Technical Interview": [
        "Tell me about yourself and your technical background.",
        "Can you briefly walk me through a technical project you've worked on recently?"
    ],
    "Situational Interview": [
        "Tell me about yourself and how you usually approach problem-solving.",
        "Imagine you’re given a new task outside your comfort zone — how would you tackle it?"
    ],
    "Competency-Based Interview": [
        "Tell me about yourself and how your experience has helped you build key professional skills.",
        "Can you share an example that highlights your ability to work in a team?"
    ],
    "Ethical or Integrity-Based Interview": [
        "Tell me about yourself and the values that guide you in your work.",
        "Have you ever faced a situation where you had to choose between doing what’s right and what’s easy? What did you do?"
    ]
}


# ------------------ storage for web sessions ------------------
WEB_SESSIONS: Dict[str, Dict[str, Any]] = {}
HISTORY_FILE = os.path.join(os.path.dirname(__file__), "history.json")


# --------------------------- CV UTILS ---------------------------
def extract_text_from_pdf(filepath):
    text = ""
    try:
        doc = fitz.open(filepath)
        for page in doc:
            text += page.get_text()
    except Exception as e:
        print("Failed to extract resume text:", e)
    return text.strip()



# --------------------------- EXPERIENCE DROPDOWN ---------------------------
def get_experience_level():
    levels = ["Fresher", "Fresher with Internship", "Work Experience"]
    print("Select your experience level:")
    for i, lvl in enumerate(levels, 1):
        print(f"{i}. {lvl}")
    while True:
        choice = input("Enter 1/2/3: ").strip()
        if choice in {"1", "2", "3"}:
            return levels[int(choice) - 1]
        else:
            print("Invalid input. Please choose 1, 2, or 3.")




# --------------------------- LANGUAGE DROPDOWN ---------------------------
def choose_language():
    levels = ["ml", "en", "kn"]
    print("Please select the language that you want to speak:")
    for i, lvl in enumerate(levels, 1):
        print(f"{i}. {lvl}")
    while True:
        choice = input("Enter 1/2/3: ").strip()
        if choice in {"1", "2", "3"}:
            return levels[int(choice) - 1]
        else:
            print("Invalid input. Please choose 1, 2, or 3.")




# --------------------------- GPT-4o UTILS ---------------------------
def generate_question_contextual(job_title, job_description, interview_type, resume_text, experience_level, history, difficulty_level):
    predefined_qs = []
    for entry in history:
        hq = entry["question"]
        if hq in sum(PREDEFINED_QUESTIONS.values(), []):
            predefined_qs.append(hq)


    context = f"You are an expert interview coach. Simulate a realistic {interview_type}. Ask level-{difficulty_level} questions that get gradually more complex. Tailor them to the candidate's job title '{job_title}'."
    if job_description:
        context += f"\nJob Description: {job_description}"
    if resume_text:
        context += f"\nResume: {resume_text[:1000]}"
    if experience_level:
        context += f"\nCandidate experience level: {experience_level}"
    if predefined_qs:
        context += f"\nAvoid repeating any of these questions: {predefined_qs}"

    context += "\nAvoid generic or off-topic questions.\nOnly output the question. Do not include any filler, lead-ins, or explanations."

    messages = [
        {"role": "system", "content": context},
    ]
    # ✅ FIX: iterate over dicts and extract keys
    for entry in history:
        q = entry.get("question")
        a = entry.get("your_answer")
        if q and a:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})

        
    messages.append({"role": "user", "content": "Ask the next question."})

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    return response.choices[0].message.content.strip()






def generate_reference_answer(question, job_title, job_description, resume_text):
    prompt = f"""
You are an AI interview coach. Generate a professional, realistic, and strong model answer to the interview question below.
This answer should be tailored to the user's background, the specific job title, and the job description.

Focus on:
- Aligning the answer with the job role and responsibilities.
- Showcasing relevant skills, experience, and achievements from the user's resume.
- Demonstrating a good cultural and motivational fit with the job.

Avoid generic responses. Make it personalized and context-aware.

Job Title: {job_title}

Job Description: {job_description}

Resume Summary: {resume_text[:1000]}

Interview Question: {question}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

     




def compare_user_to_model_answer(user_answer, model_answer, job_title, job_description, resume_text):
    prompt = f"""
You are an AI interview coach. Analyze the user's answer to a job interview question in detail. 
Your goal is to provide clear, structured feedback without rewriting or revising the user's answer.

Focus on the following points only:
1. How well the user answered the question.
2. What mistakes or gaps are present in the answer.
3. How the user can improve their answer in future interviews.

Use the model answer only as a reference point. Do not include any revised or rewritten version of the user's answer.

Job Title: {job_title}

Job Description: {job_description}

Resume Summary: {resume_text[:1000]}

Model Answer (for reference): {model_answer}

User's Answer: {user_answer}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()



# --------------------------- TTS ---------------------------
def speak_text(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)
    engine.say(text)
    engine.runAndWait()



# --------------------------- AUDIO RECORDING ---------------------------
def record_audio(filename="response.wav"):
    mode = input("Choose recording mode - type '1' for timed or '2' for manual stop: ").strip()
    while mode not in ['1', '2']:
        mode = input("Invalid. Choose '1' or '2': ").strip()

    samplerate = 16000
    if mode == '1':
        while True:
            try:
                duration = float(input("Enter recording duration in seconds: "))
                break
            except ValueError:
                print("Invalid input. Try again.")
        print(f"Recording for {duration} seconds...")
        audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
        sd.wait()
    else:
        print("Recording... Press ENTER to stop.")
        q = queue.Queue()

        def callback(indata, frames, time_, status):
            q.put(indata.copy())

        with sd.InputStream(samplerate=samplerate, channels=1, callback=callback):
            frames = []
            stopper = threading.Thread(target=input, args=("Press ENTER to stop...",))
            stopper.start()
            while stopper.is_alive():
                frames.append(q.get())
            audio = np.concatenate(frames)

    wav.write(filename, samplerate, (audio * 32767).astype(np.int16))
    return filename




    # ---------------------- Noise Reduction ----------------------
def reduce_noise(input_file):
    y, sr = librosa.load(input_file, sr=16000)
    reduced = nr.reduce_noise(y=y, sr=sr)
    return reduced, sr

# ---------------------- Dereverberation ----------------------
def dereverb_audio(y, sr):
    y = np.expand_dims(y, axis=0)  # Shape: [n_channels, n_samples]
    dereverb = wpe.wpe(y)[0]
    return dereverb

# ---------------------- Bandpass Filter ----------------------
def bandpass_filter(audio, sr, lowcut=300.0, highcut=3400.0, order=4):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_audio = lfilter(b, a, audio)
    return filtered_audio

# ---------------------- Volume Normalization ----------------------
def normalize_audio(y, sr):
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(y)
    normalized = pyln.normalize.loudness(y, loudness, -23.0)  # Target loudness: -23 LUFS
    return normalized

# ---------------------- Voice Enhancement ----------------------
def load_custom_speech_enhancer(model_dir):
    with open(f"{model_dir}/hyperparams.yaml") as f:
        hparams = load_hyperpyyaml(f)

    model = hparams['modules']['enhance_model']
    state_dict = torch.load(f"{model_dir}/enhance_model.ckpt", map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    return model

def enhance_audio(input_file, output_file, model_dir):
    audio, sr = torchaudio.load(input_file)
    if sr != 16000:
        audio = torchaudio.transforms.Resample(sr, 16000)(audio)

    model = load_custom_speech_enhancer(model_dir)
    model.eval()

    with torch.no_grad():
        enhanced_tuple = model(audio)
        enhanced_audio = enhanced_tuple[0]  # Extract the enhanced speech
    sf.write(output_file, enhanced_audio.squeeze().cpu().numpy(), 16000)
    return output_file




# ---------------------- Full Preprocessing Pipeline ----------------------
def preprocess_audio_pipeline(voice_file, model_dir="C:/code_projects/RP2/pretrained_models/enhance"):
    print("🔊 Step 1: Noise Reduction...")
    y, sr = reduce_noise(voice_file)

    print("🎤 Step 2: Dereverberation...")
    y = dereverb_audio(y, sr)

    print("🎚 Step 3: Bandpass Filtering...")
    y = bandpass_filter(y, sr)

    print("📢 Step 4: Volume Normalization...")
    y = normalize_audio(y, sr)

    # Save intermediate cleaned audio
    temp_file = "temp_cleaned.wav"
    sf.write(temp_file, y, sr)

    print("🤖 Step 5: Voice Enhancement (Deep Learning)...")
    final_output = "voice_after_cleaning.wav"
    enhance_audio(temp_file, final_output, model_dir)

    print(f"✅ Preprocessing complete. Cleaned file saved at: {final_output}")
    return final_output





# -----------------------------
# Utility: Extract First 15 Seconds
# -----------------------------
""" avoids leading silence.

Signature preserved:
    extract_first_15s(audio_path, output_path="temp_15s.wav", duration=15)

Behavior preserved:
- If input is <= duration seconds, return the original `audio_path` (no write).
Requires: librosa, soundfile, numpy
"""


def extract_first_15s(audio_path, output_path="temp_15s.wav", duration=15):
    """Extract ~`duration` seconds containing active speech; avoid leading silence.

    Returns the written `output_path` on success; if the file is already short,
    returns the original `audio_path`.
    """

    # Fixed defaults (tune here if needed without changing the function signature).
    SR = 16_000
    TOP_DB = 30.0
    FRAME_LENGTH = 2048
    HOP_LENGTH = 512
    MERGE_GAP_S = 0.30  # why: avoid splitting words by tiny pauses

    # Load mono audio at target sample rate.
    y, sr = librosa.load(audio_path, sr=SR, mono=True)

    # Preserve original behavior for short inputs.
    if librosa.get_duration(y=y, sr=sr) <= float(duration):
        return audio_path

    win_len = int(round(float(duration) * sr))
    total_len = int(y.size)

    # 1) Non-silent intervals (sample indices).
    intervals = librosa.effects.split(
        y=y,
        top_db=TOP_DB,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
    )

    # 2) Merge intervals separated by < MERGE_GAP_S seconds.
    if intervals.size:
        min_gap = int(MERGE_GAP_S * sr)
        merged = []
        s0, e0 = int(intervals[0, 0]), int(intervals[0, 1])
        for s, e in intervals[1:]:
            s, e = int(s), int(e)
            if s - e0 <= min_gap:
                e0 = e
            else:
                merged.append([s0, e0])
                s0, e0 = s, e
        merged.append([s0, e0])
        intervals = np.asarray(merged, dtype=int)

    # 3) Build frame-level activity mask.
    n_frames = int(np.ceil(total_len / HOP_LENGTH))
    mask = np.zeros(n_frames, dtype=np.float32)
    for s, e in intervals:
        fs = max(0, int(s) // HOP_LENGTH)
        fe = min(n_frames, int(np.ceil(int(e) / HOP_LENGTH)))
        mask[fs:fe] = 1.0

    window_frames = int(np.ceil(win_len / HOP_LENGTH))

    # 4) Choose start frame: maximize activity; fallback to highest RMS.
    if mask.sum() > 0 and n_frames > window_frames:
        kernel = np.ones(window_frames, dtype=np.float32)
        coverage = np.convolve(mask, kernel, mode="valid")
        start_frame = int(np.argmax(coverage))
    else:
        rms = librosa.feature.rms(y=y, frame_length=2 * HOP_LENGTH, hop_length=HOP_LENGTH, center=True)[0]
        if rms.size > window_frames:
            kernel = np.ones(window_frames, dtype=np.float32)
            energy = np.convolve(rms, kernel, mode="valid")
            start_frame = int(np.argmax(energy))
        else:
            start_frame = 0

    start = min(int(start_frame * HOP_LENGTH), total_len - win_len)
    end = start + win_len

    segment = y[start:end]

    # Pad rare shortfall due to rounding.
    if segment.size < win_len:
        import numpy as _np
        segment = _np.pad(segment, (0, win_len - segment.size), mode="constant")

    sf.write(output_path, segment, sr)
    return output_path


# -----------------------------
# Whisper Language Detection
# -----------------------------
def detect_language_whisper(audio_path):
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(MODEL.device)
    _, probs = MODEL.detect_language(mel)
    detected_lang = max(probs, key=probs.get)
    confidence = probs[detected_lang]
    print(f"🎯 Whisper Detected Language: {detected_lang}, Confidence: {confidence:.2f}")
    return detected_lang, confidence


# -----------------------------
# GPT Fallback Language Verification
# -----------------------------
def verify_language_with_gpt(audio_path):
    # Step 1: Transcribe audio
    with open(audio_path, "rb") as f:
        transcript_response = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=f,
        )
    transcript = transcript_response.text.strip()

    # Step 2: Use chat model to detect language code
    chat_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a language detector."},
            {"role": "user", "content": f"Detect the language of the following text and return ONLY the lowercase language code: en, ml, kn, or other. Output must be exactly one of these. Text: {transcript}"}
        ]
    )

    detected_lang = chat_response.choices[0].message.content.strip().lower()
    print(f"🤖 GPT Verified Language: {detected_lang}")
    return detected_lang



def transcribe_with_gpt(audio_path, detected_language):
    lang_map = {"en": "English", "ml": "Malayalam", "kn": "Kannada"}
    language_name = lang_map.get(detected_language, "English")

    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",  # ✅ Upgraded model
            file=f,
            prompt=(
                f"The speaker is talking in {language_name}. "
                "Transcribe the audio exactly as spoken, strictly preserving:\n"
                "- All filler words (e.g., 'um', 'uh').\n"
                "- All hesitations & incomplete phrases.\n"
                "- All pauses as '...'.\n"
                "Do NOT clean grammar or remove natural speech patterns."
            )
        )
    return response.text.strip()

# -----------------------------
# MFA Alignment (Word-level Timing, Pauses)
# -----------------------------


def run_mfa_alignment(audio_path, transcript_text, acoustic_model):
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    corpus_dir = "mfa_corpus_temp"
    aligned_dir = "aligned_output"

    # 🧹 Clean up any existing data
    if os.path.exists(corpus_dir):
        shutil.rmtree(corpus_dir)
    if os.path.exists(aligned_dir):
        shutil.rmtree(aligned_dir)

    os.makedirs(corpus_dir, exist_ok=True)
    os.makedirs(aligned_dir, exist_ok=True)

    # Prepare corpus
    shutil.copy(audio_path, os.path.join(corpus_dir, f"{base_name}.wav"))

    # Split transcript into sentences for MFA
    sentences = re.split(r'(?<=[.!?])\s+', transcript_text.strip())
    
    with open(os.path.join(corpus_dir, f"{base_name}.lab"), "w", encoding="utf-8") as f:
        for sent in sentences:
            if sent.strip():
                f.write(sent.strip() + "\n")
    with open(os.path.join(corpus_dir, f"{base_name}.lab"), "w", encoding="utf-8") as f:
        f.write(transcript_text)

    # Build MFA command (subprocess)
    cmd = [
        r"C:\code_projects\MFA\Scripts\mfa.exe",
        "align",
        "--clean",
        corpus_dir,
        acoustic_model,
        acoustic_model,
        aligned_dir,
        "--single_speaker",
        "--g2p",
        "--beam", "500",
        "--retry_beam", "1000",
        "--output_format", "json"
    ]

    print("🔧 Running MFA alignment...")
    subprocess.run(cmd, check=True)

    # Return alignment JSON path
    for file in os.listdir(aligned_dir):
        if file.endswith(".json"):
            return os.path.join(aligned_dir, file)

    return None



# -----------------------------
# Main Transcription Pipeline
# -----------------------------
def transcribe_audio_pipeline_for_record(audio_path, language):
   

    # GPT-4 transcription
    native_text = transcribe_with_gpt(audio_path, language)

    # Translation if non-English
    if language in ["ml", "kn"]:
        english_translation = GoogleTranslator(source="auto", target="en").translate(native_text)
    else:
        english_translation = native_text

    return {
        "language": language,
        "native_text": native_text,
        "english_translation": english_translation
    }



# -----------------------------
# Main Pipeline with Loop Logic
# -----------------------------


def transcribe_audio_pipeline(voice_after_cleaning):
    mismatch_count = 0

    while True:
        # Step 1: Extract first 15 seconds for language identification
        clip_path = extract_first_15s(voice_after_cleaning)

        # Step 2: Language detection via Whisper
        whisper_lang, _ = detect_language_whisper(clip_path)

        # Step 3: Language verification via GPT
        gpt_lang = verify_language_with_gpt(clip_path)

        # Step 4: Matching logic and validation
        if whisper_lang == gpt_lang:
            if whisper_lang in SUPPORTED_LANGUAGES:
                break  # Both match and supported, continue
            else:
                print("⚠ Language detected twice, but it's unsupported. Please re-record your voice.")
                audio_path = record_audio()
                voice_after_cleaning = preprocess_audio_pipeline(audio_path) 
                mismatch_count = 0
                continue
        else:
            mismatch_count += 1
            if mismatch_count >= 2:
                print("🎤 Whisper and GPT mismatch repeated twice. Please choose your language manually.")
                whisper_lang = choose_language()
                break
            print("🎤 Whisper and GPT disagree. Please record clearly.")
            audio_path = record_audio()
            voice_after_cleaning = preprocess_audio_pipeline(audio_path) 
            continue

    # Step 5: Full transcription using GPT
    native_text = transcribe_with_gpt(voice_after_cleaning, whisper_lang)

    # Step 6: Translation to English if necessary
    if whisper_lang in ["ml", "kn"]:
        english_translation = GoogleTranslator(source="auto", target="en").translate(native_text)
    else:
        english_translation = native_text

    return {
        "language": whisper_lang,
        "native_text": native_text,
        "english_translation": english_translation
    }






# -----------------------------
# 1. MFA Pause Detection (Phones Tier)
# -----------------------------
def detect_pauses_from_mfa(mfa_json, min_pause=0.8):
    pauses = []
    with open(mfa_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    phones = data["tiers"]["phones"]["entries"]

    for start, end, label in phones:
        if label == "sil":  # Explicit silence from MFA
            duration = end - start
            if duration >= min_pause:
                pauses.append(duration)
    return pauses



def filler_hesitation(json_path: str):

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
  
    # MFA phones = entire phoneme sequence (not just inside words)
    filler_phonemes = []
    
    for phone in data.get("phones", []):
        phoneme = phone["phone"].split("_")[0]  # Strip stress
        if phoneme in {p.strip("012") for p in FILLER_PHONEMES}:
            filler_phonemes.append({
                "phoneme": phoneme,
                "start": phone["start"],
                "end": phone["end"]
            })

    return filler_phonemes


# -----------------------------
# Analyze Audio (Enhanced with MFA)
# -----------------------------

def analyze_audio(filename, transcript_data):
    """
    STRICT language behavior:
      - Uses ONLY the user's spoken language (en/ml/kn) for MFA + filler logic.
      - Requires a native transcript in that language.
      - Will NOT fall back to English/translation if native text is missing.

    Expected transcript_data (any one of these is OK):
      {"native_text": <str>, "language": "en"|"ml"|"kn"}
      {"native": <str>, "language": "en"|"ml"|"kn"}

    Returns metrics with both snake_case and camelCase keys to match the UI.
    """
    # ---- validate transcript data strictly ----
    if not isinstance(transcript_data, dict):
        raise ValueError("transcript_data must be a dict with native text and language")

    detected_lang = transcript_data.get("language") or transcript_data.get("detectedLang")
    if detected_lang not in {"en", "ml", "kn"}:
        raise ValueError("Missing or unsupported language for voice analysis (expected 'en','ml','kn')")

    native_text = transcript_data.get("native_text") or transcript_data.get("native")
    if not native_text or not isinstance(native_text, str) or not native_text.strip():
        # IMPORTANT: do NOT fall back to english/english_translation here
        raise ValueError("Missing native transcript for voice analysis")

    # ---- pick acoustic model strictly by language ----
    model_map = {
        "en": "english_mfa",  # Indian English works better with this model
        "ml": "tamil_cv",     # Dravidian phonetic closeness (placeholder you already use)
        "kn": "tamil_cv"
    }
    acoustic_model = model_map[detected_lang]

    # ---- MFA alignment ----
    alignment_file = run_mfa_alignment(filename, native_text, acoustic_model)

    # ---- filler list strictly by language ----
    if detected_lang == "ml":
        filler_list = FULL_FILLER_WORDS_ML
    elif detected_lang == "kn":
        filler_list = FULL_FILLER_WORDS_KN
    else:
        filler_list = FULL_FILLER_WORDS_EN

    # ---- audio metrics ----
    y, sr = librosa.load(filename)
    duration = float(librosa.get_duration(y=y, sr=sr)) if sr else 0.0
    wpm = 0.0
    if duration > 0:
        wpm = len(native_text.split()) / (duration / 60.0)

    # ---- filler detection ----
    words = native_text.lower().split()
    filler_words = [w for w in words if w in filler_list]
    filler_ratio = (len(filler_words) / max(1, len(words))) if words else 0.0

    # ---- pauses from MFA ----
    pauses = detect_pauses_from_mfa(alignment_file)

    # ---- scoring (unchanged logic) ----
    score = 100
    if wpm < 90 or wpm > 180:
        score -= 15
    if filler_ratio > 0.05:
        score -= 20
    if len(native_text.strip()) < 5:
        score -= 50
    if len(pauses) > 5:
        score -= 20

    # Return both snake_case and camelCase to keep the UI happy
    return {
        "duration_sec": duration,
        "wpm": wpm,
        "filler_ratio": filler_ratio,
        "fillerRatio": filler_ratio,
        "filler_count": len(filler_words),
        "fillerCount": len(filler_words),
        "filler_words": filler_words,
        "pause_count": len(pauses),
        "pauseCount": len(pauses),
        "Duration_of_pause": pauses,
        "durationOfPause": pauses,
        "score": max(0, score),
        "language": detected_lang
    }




# --------------------------- INTERVIEW LOOP ---------------------------
def start_interview(job_title, job_description, interview_type, resume_text, experience_level, manual_type=False):
    history = []
    difficulty_levels = ["basic", "intermediate", "advanced"]
    level_index = 0
    reference_answer_index = 0
    predefined_mode = not manual_type and interview_type in PREDEFINED_QUESTIONS
    predefined_qs = PREDEFINED_QUESTIONS.get(interview_type, [])

    print("\nInterview started. Press Enter when prompted to continue or stop.\n")
    while True:
        try:
            # Question selection
            if predefined_mode and level_index < len(predefined_qs):
                question = predefined_qs[level_index]
            elif manual_type and level_index == 0:
                question = "Tell me about yourself."
            else:
                difficulty = difficulty_levels[min(level_index - (2 if predefined_mode else 1), len(difficulty_levels) - 1)]
                question = generate_question_contextual(job_title, job_description, interview_type, resume_text, experience_level, history, difficulty)

            print("\nQ:", question)
            level_index = level_index + 1
            hear_q = input("Do you want to hear this question? (yes/no): ").strip().lower()
            if hear_q == 'yes':
                speak_text(question)

            # Record answer
            filename = record_audio()
            voice_after_cleaning = preprocess_audio_pipeline(filename) 

            # ✅ Get transcription (native + translated)
            transcript_data = transcribe_audio_pipeline(voice_after_cleaning)
            native_text = transcript_data["native_text"]
            translated_text = transcript_data["english_translation"]
            print(f"\nYour Actual Answer (Native: {transcript_data['language']}): {native_text}")
            print(f"\nYour Answer (English: {transcript_data['language']}): {translated_text}")
            

            playback = input("Do you want to hear your recorded response? (yes/no): ").strip().lower()
            if playback == 'yes':
                display(ipd.Audio(filename))
                display(ipd.Audio(voice_after_cleaning))

            

            # ✅ GPT Feedback using translated text
            compare_now = input("Do you want real-time AI feedback on your answer? (yes/no): ").strip().lower()
            if compare_now == 'yes':
                reference_answer_index = reference_answer_index + 1
                # Generate reference answer
                reference = generate_reference_answer(question, job_title, job_description, resume_text)
                comparison = compare_user_to_model_answer(
                    translated_text,  # ✅ Use English translation
                    reference,
                    job_title,
                    job_description,
                    resume_text
                )
                print("\n--- AI FEEDBACK ---\n", comparison)

            # ✅ Voice Analysis using native transcript
            see_analysis = input("Do you want to see voice analysis (WPM, filler, score)? (yes/no): ").strip().lower()
            if see_analysis == 'yes':
                feedback = analyze_audio(voice_after_cleaning, transcript_data)  # ✅ Native text analyzed
                print("\n--- VOICE FEEDBACK ---")
                print(f"Speaking Rate (WPM): {feedback['wpm']:.2f}")
                print(f"Filler Ratio: {feedback['filler_ratio']:.2%}")
                print(f"Score: {feedback['score']}/100")
                print(f"filler_count: {feedback['filler_count']:}")
                print(f"pause_count: {feedback['pause_count']:}")
                print(f"Duration_of_pause: {feedback['Duration_of_pause']}")
              
                

            # Show reference answer (model)
            see_model = input("Do you want to see the model reference answer? (yes/no): ").strip().lower()
            if see_model == 'yes':
                if reference_answer_index == level_index:
                    print("\nReference Answer:\n", reference)
                else:
                    reference = generate_reference_answer(question, job_title, job_description, resume_text)
                    print("\nReference Answer:\n", reference)
                speak = input("Do you want to hear it? (yes/no): ").strip().lower()
                if speak == 'yes':
                    speak_text(reference)

            # ✅ Append only required fields
            history.append({
                "index": level_index,
                "question": question,
                "your_answer": native_text
            })
            reference_answer_index = level_index

            # ✅ Save to JSON after every loop
            with open("history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=4)
            

            # Loop control
            next_q = input("\nPress Enter to continue or type 'stop' to end: ").strip().lower()
            if next_q == 'stop':
                break

        except KeyboardInterrupt:
            print("\nInterview ended by user.")
            break




# --------------------------- RECORDED SESSION LOOP ---------------------------
def record_with_retry(question_text, filename_prefix):
    attempts = []
    final_path = None
    counter = 1

    while True:
        print(f"\nRecording Attempt #{counter} for: {question_text}")
        filename = f"{filename_prefix}_try{counter}.wav"
        record_audio(filename)

        display(ipd.Audio(filename))

        while True:
            retry = input("Do you want to proceed with this answer? (yes to keep, no to re-record): ").strip().lower()
            if retry in ['yes', 'no']:
                break
        
        if retry == 'yes':
            final_path = filename
            


            break
        else:
            attempts.append(filename)
            counter += 1

    for old in attempts:
        if os.path.exists(old):
            os.remove(old)

    return final_path



# --------------------------- RECORDED INTERVIEW ---------------------------
def start_recorded_session(job_title, job_description, interview_type, resume_text, experience_level, manual_type=False):
    lang = choose_language()
    history = []
    difficulty_levels = ["basic", "intermediate", "advanced"]
    level_index = 0
    reference_answer_index = 0
    predefined_mode = not manual_type and interview_type in PREDEFINED_QUESTIONS
    predefined_qs = PREDEFINED_QUESTIONS.get(interview_type, [])

    print("\nRecorded Session Interview Started\n")
    while True:
        try:
            if predefined_mode and level_index < len(predefined_qs):
                question = predefined_qs[level_index]
            elif manual_type and level_index == 0:
                question = "Tell me about yourself."
            else:
                difficulty = difficulty_levels[min(level_index - (2 if predefined_mode else 1), len(difficulty_levels) - 1)]
                question = generate_question_contextual(job_title, job_description, interview_type, resume_text, experience_level, history, difficulty)

            print("\nQ:", question)
            level_index = level_index + 1

            while True:
                hear_q = input("Do you want to hear this question? (yes/no): ").strip().lower()
                if hear_q in ['yes', 'no']:
                    break
            
            if hear_q == 'yes':
                speak_text(question)

            final_audio = record_with_retry(question, f"answer_{level_index+1}")
            voice_after_cleaning = preprocess_audio_pipeline(final_audio) 
            display(ipd.Audio(voice_after_cleaning))

            # ✅ Get transcription (native + translated)
            transcript_data = transcribe_audio_pipeline_for_record(voice_after_cleaning,lang)
            native_text = transcript_data["native_text"]
            translated_text = transcript_data["english_translation"]
            print(f"\nYour Original Answer (In Native: {transcript_data['language']}): {native_text}")
            print(f"\nYour Translated Answer (English: {transcript_data['language']}): {translated_text}")
            
            

            while True:
                compare_now = input("Do you want real-time AI feedback on your answer? (yes/no): ").strip().lower()
                if compare_now in ['yes', 'no']:
                    break
            
            if compare_now == 'yes':
                reference_answer_index = reference_answer_index + 1
                reference = generate_reference_answer(question, job_title, job_description, resume_text)
                comparison = compare_user_to_model_answer(translated_text, reference, job_title, job_description, resume_text)
                print("\n--- AI FEEDBACK ---\n", comparison)

            while True:
                see_analysis = input("Do you want to see voice analysis (WPM, filler, score)? (yes/no): ").strip().lower()
                if see_analysis in ['yes', 'no']:
                    break
            
            if see_analysis == 'yes':
                feedback = analyze_audio(voice_after_cleaning, transcript_data)
                print("\n--- VOICE FEEDBACK ---")
                print(f"Speaking Rate (WPM): {feedback['wpm']:.2f}")
                print(f"Filler Ratio: {feedback['filler_ratio']:.2%}")
                print(f"Score: {feedback['score']}/100")
                print(f"filler_count: {feedback['filler_count']}")
                print(f"pause_count: {feedback['pause_count']}")
                print(f"Duration_of_pause: {feedback['Duration_of_pause']}")
                
                
                

            while True:
                see_model = input("Do you want to see the model reference answer? (yes/no): ").strip().lower()
                if see_model in ['yes', 'no']:
                    break           
            if see_model == 'yes':
                if reference_answer_index == level_index:
                    print("\nReference Answer:\n", reference)
                else:
                    reference = generate_reference_answer(question, job_title, job_description, resume_text)
                    print("\nReference Answer:\n", reference)

                while True:
                    speak = input("Do you want to hear it? (yes/no): ").strip().lower()
                    if speak in ['yes', 'no']:
                        break
                if speak == 'yes':
                    speak_text(reference)
                    

            # ✅ Append only required fields
            history.append({
                "index": level_index,
                "question": question,
                "your_answer": native_text
            })
            reference_answer_index = level_index

            # ✅ Save to JSON after every loop
            with open("history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=4)
            

            next_q = input("\nPress Enter to continue or type 'stop' to end: ").strip().lower()
            if next_q == 'stop':
                break
        except KeyboardInterrupt:
            print("\nInterview ended by user.")
            break




# ... existing imports and functions above ...

# =====================================================================
# Web API helpers
#
# The original start_interview() function uses input() and thus
# blocks when called from Flask. The following wrappers keep
# session state and do not call input().

_WEB_SESSIONS = {}

class WebInterviewSession:
    """Holds state for a single interview session in the web context."""
    def __init__(self, job_title, job_description, interview_type,
                 resume_input, experience_level, manual_type):
        self.job_title = job_title
        self.job_description = job_description
        self.interview_type = interview_type
        self.resume_input = resume_input  # resume text or path
        self.experience_level = experience_level
        self.manual_type = manual_type
        self.history = []
        self.level_index = 0
        self.predefined_mode = (not manual_type) and (interview_type in PREDEFINED_QUESTIONS)
        self.predefined_qs = PREDEFINED_QUESTIONS.get(interview_type, [])
        self.mismatch_count = 0

    def next_question(self):
        # Use predefined questions when available
        if self.predefined_mode and self.level_index < len(self.predefined_qs):
            question = self.predefined_qs[self.level_index]
        elif self.manual_type and self.level_index == 0:
            question = "Tell me about yourself."
        else:
            # Otherwise call your GPT-based generator with difficulty escalation
            difficulty_levels = ["basic", "intermediate", "advanced"]
            offset = 2 if self.predefined_mode else 1
            diff_index = max(0, min(self.level_index - offset,
                                    len(difficulty_levels) - 1))
            difficulty = difficulty_levels[diff_index]
            question = generate_question_contextual(
                self.job_title,
                self.job_description,
                self.interview_type,
                self.resume_input,
                self.experience_level,
                self.history,
                difficulty
            )
        self.level_index += 1
        return str(question)

# --------------------------- start: web ------------------------
def start_interview_web(job_title: str,
                        job_description: str,
                        interview_type: str,
                        resume_input: str,
                        experience_level: str,
                        manual_type: bool) -> Dict[str, Any]:
    """
    Create a web session and return the first question.
    manual_type=True  -> user typed interview type: start with generic "Tell me about yourself."
    manual_type=False -> user picked from list: ask both predefined questions first, then GPT.
    """
    # Normalize experience_level codes into readable labels
    exp_map = {
        "1": "Fresher",
        "2": "Fresher with Internship",
        "3": "Work Experience",
    }
    experience_level = exp_map.get(experience_level, experience_level)

    sid = uuid.uuid4().hex

    # Prepare initial question sequence
    if manual_type:
        q_sequence = ["Tell me about yourself."]
    else:
        q_sequence = PREDEFINED_QUESTIONS.get(interview_type, [])[:]

    # 🔹 Extract resume text once (if a PDF path was supplied)
    resume_text = ""
    try:
        if resume_input:
            if os.path.isfile(resume_input):
                resume_text = extract_text_from_pdf(resume_input)
            else:
                resume_text = str(resume_input)
    except Exception:
        resume_text = ""

    # 🔹 Store session metadata
    WEB_SESSIONS[sid] = {
        "job_title": job_title,
        "job_description": job_description,
        "job_desc": job_description,  # alias; harmless convenience
        "interview_type": interview_type,
        "resume_input": resume_input,  # keep path (or text, if you prefer)
        "experience_level": experience_level,
        "manual_type": manual_type,

        # question sequencing
        "q_idx": 0,
        "predefined": (not manual_type),
        "q_sequence": q_sequence,   # consumed first; then GPT questions afterwards

        # language / audio step state
        "mismatch_count": 0,
        "override_language": None,
        "last_cleaned_path": "",
        "history": [],  # [{question, native_text, english_translation, language}]

        # 🔹 new fields
        "resume_text": resume_text,
        "level_index": 0
    }

    # 🔹 Handle first question and update session state
    if q_sequence:
        first_q = q_sequence[0]
        WEB_SESSIONS[sid]["level_index"] = 1
        WEB_SESSIONS[sid]["last_question"] = first_q
    else:
        # if no predefined questions, call contextual generator
        first_q = generate_question_contextual(
            job_title, job_description, interview_type, resume_text,
            experience_level, [], "basic")
        WEB_SESSIONS[sid]["level_index"] = 1
        WEB_SESSIONS[sid]["last_question"] = first_q

    return {"session_id": sid, "question": first_q}




def _append_and_save_history(sid: str, item: Dict[str, Any]):
    sess = WEB_SESSIONS.get(sid)
    if not sess:
        return
    sess["history"].append({
        "ts": time.time(),
        **item
    })
    # dump minimal history to history.json
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump({sid: sess["history"]}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def save_history_json_web(session_id: str):
    """Expose a callable if the Flask app wants to force-save."""
    sess = WEB_SESSIONS.get(session_id)
    if not sess:
        return
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump({session_id: sess["history"]}, f, ensure_ascii=False, indent=2)



# ---- helper: call transcribe_audio_pipeline regardless of signature ----
def _transcribe_with_best_effort(audio_path: str, lang: str):
    """
    Try several common signatures of transcribe_audio_pipeline so we don't crash
    if the notebook version doesn't expose 'language_hint'.
    """
    try:
        # Preferred if your function supports a 'language' kw
        return transcribe_audio_pipeline(audio_path, language=lang)
    except TypeError:
        pass
    try:
        # What I tried first in the web wrapper
        return transcribe_audio_pipeline(audio_path, language_hint=lang)
    except TypeError:
        pass
    try:
        # Positional language
        return transcribe_audio_pipeline(audio_path, lang)
    except TypeError:
        pass
    # No language argument supported – just run it
    return transcribe_audio_pipeline(audio_path)


def process_answer_web(session_id: str, audio_path: str, manual_language: str = None) -> Dict[str, Any]:
    """
    Web answer pipeline:
      1) 5-step preprocess -> cleaned path
      2) Language detect (Whisper + GPT) unless manual_language is provided
         - single mismatch -> ask to re-record (early return)
         - second mismatch -> ask for manual language (early return)
         - unsupported but matched -> ask to re-record (early return)
      3) Transcribe exactly once (no second language pass)
      4) Translate to English when needed
      5) Append to history
      6) (Optional) AI feedback + model answer
    """
    sess = WEB_SESSIONS.get(session_id)
    assert sess is not None, "Invalid session"

    # ---- 1) Preprocess up-front so `cleaned` ALWAYS exists ----
    cleaned = preprocess_audio_pipeline(audio_path)
    sess["last_cleaned_path"] = cleaned

    # ---- 2) Language selection (one pass) ----
    lang = manual_language or sess.get("override_language")
    if not lang:
        # detect on the cleaned audio's active first segment
        clip_path = extract_first_15s(cleaned)
        whisper_lang, _ = detect_language_whisper(clip_path)
        gpt_lang = verify_language_with_gpt(clip_path)

        if whisper_lang == gpt_lang:
            if whisper_lang in SUPPORTED_LANGUAGES:
                lang = whisper_lang
                sess["mismatch_count"] = 0
            else:
                # matched, but unsupported -> ask user to re-record
                sess["mismatch_count"] = 0
                return {
                    "mismatch": True,
                    "mismatchMessage": (
                        "⚠ Language detected, but it's unsupported. "
                        "Supported languages are ENGLISH, MALAYALAM, KANNADA. Please re-record."
                    ),
                    "langs": [whisper_lang, gpt_lang],
                    "cleaned_audio_path": cleaned
                }
        else:
            # mismatch between Whisper and GPT
            sess["mismatch_count"] = sess.get("mismatch_count", 0) + 1
            if sess["mismatch_count"] >= 2:
                return {
                    "mismatch": True,
                    "mismatchMessage": "🎤 Whisper and GPT mismatched twice. Please choose your language manually.",
                    "manualLanguageNeeded": True,
                    "langs": [whisper_lang, gpt_lang],
                    "cleaned_audio_path": cleaned
                }
            else:
                return {
                    "mismatch": True,
                    "mismatchMessage": "🎤 Whisper and GPT disagree. Please re-record clearly.",
                    "langs": [whisper_lang, gpt_lang],
                    "cleaned_audio_path": cleaned
                }

    # If user forced a language (manual selection), keep it for this answer
    if manual_language:
        sess["override_language"] = manual_language
        lang = manual_language

    # ---- 3) Transcribe exactly once (no extra language detection) ----
    native_text = ""
    try:
        # preferred: your direct GPT transcription that respects `lang`
        native_text = transcribe_with_gpt(cleaned, lang)
    except Exception:
        # fallback: your notebook/web recorder helper signature
        try:
            tmp = transcribe_audio_pipeline_for_record(cleaned, lang)
            native_text = tmp.get("native_text", "") if isinstance(tmp, dict) else str(tmp or "")
        except Exception:
            # last resort: plain transcribe function (may return str or dict)
            tmp = transcribe_audio_pipeline(cleaned)
            native_text = tmp.get("native_text", "") if isinstance(tmp, dict) else str(tmp or "")

    # ---- 4) Translate to English only for display; voice analysis will use native ----
    if lang == "en":
        english_translation = native_text
    else:
        english_translation = GoogleTranslator(source="auto", target="en").translate(native_text)

    # ---- 5) Append to history ----
    _append_and_save_history(session_id, {
        "question": current_question_web(session_id),
        "language": lang,
        "native_text": native_text,
        "english_translation": english_translation
    })


    return {
        "mismatch": False,
        "mismatchMessage": "",
        "language": lang,
        "native_text": native_text,
        "english_translation": english_translation,
        "cleaned_audio_path": cleaned
    }

    


def finalize_with_manual_language_web(session_id: str, audio_path: str, language: str) -> Dict[str, Any]:
    """
    Called by /manual_language after two mismatches; continues without re-recording.
    """
    return process_answer_web(session_id, audio_path, manual_language=language)

# ---------------- Next question sequencing ---------------------
def current_question_web(session_id: str) -> str:
    sess = WEB_SESSIONS.get(session_id)
    if not sess:
        return ""
    qseq = sess.get("q_sequence", [])
    idx  = sess.get("q_idx", 0)
    if idx < len(qseq):
        return qseq[idx]
    # beyond predefined: this is the one we just generated previously for the UI
    return sess.get("last_question", "")


def next_question_web(session_id: str) -> str:
    sess = WEB_SESSIONS.get(session_id)
    assert sess is not None, "Invalid session"
    qseq = sess.get("q_sequence", [])
    idx  = sess.get("q_idx", 0)
    level_index = sess.get("level_index", 0)

    # Serve predefined questions while available
    if idx + 1 < len(qseq):
        sess["q_idx"] = idx + 1
        sess["level_index"] = level_index + 1
        q = qseq[sess["q_idx"]]
        sess["last_question"] = q
        return q

    # No more predefined: compute difficulty
    predefined_count = len(qseq) if len(qseq) > 0 else 1
    difficulty_levels = ["basic","intermediate","advanced"]
    diff_index = level_index - predefined_count
    if diff_index < 0: diff_index = 0
    if diff_index >= len(difficulty_levels): diff_index = len(difficulty_levels)-1
    difficulty = difficulty_levels[diff_index]

    # Build history for GPT from session['history']
    history_for_gpt = []
    for entry in sess.get("history", []):
        q_text = entry.get("question")
        ans = entry.get("english_translation") or entry.get("native_text")
        if q_text and ans:
            history_for_gpt.append({"question": q_text, "your_answer": ans})

    resume_text = sess.get("resume_text") or ""

    # Always use contextual generator (no fallback)
    q = generate_question_contextual(
        sess["job_title"],
        sess.get("job_description") or sess.get("job_desc") or "",
        sess["interview_type"],
        resume_text,
        sess.get("experience_level",""),
        history_for_gpt,
        difficulty
    )

    sess["level_index"] = level_index + 1
    sess["q_idx"] = max(len(qseq), idx + 1)
    sess["last_question"] = q
    return q





# ---------------- Voice analysis (MFA) only on demand ----------
def analyze_audio_web(session_id: str, audio_path: str, transcript_data: dict) -> dict:
    # If your low-level analyze_audio requires transcript_data, pass it.
    if "analyze_audio" in globals():
        try:
            return analyze_audio(audio_path, transcript_data)
        except TypeError:
            # Older signature without transcript
            return analyze_audio(audio_path)
    return {}





def generate_ai_feedback_web(session_id: str) -> Dict[str, Any]:
    sess = WEB_SESSIONS.get(session_id)
    if not sess or not sess.get("history"):
        return {"ai_feedback": ""}
    last = sess["history"][-1]
    user_answer_en = last.get("english_translation") or last.get("native_text") or ""
    question = last.get("question") or sess.get("last_question") or ""
    job_title = sess.get("job_title","")
    job_desc  = sess.get("job_description") or sess.get("job_desc") or ""
    resume_text = sess.get("resume_text") or sess.get("resume_input") or ""
    try:
        model_ans = generate_reference_answer(question, job_title, job_desc, resume_text)
        feedback  = compare_user_to_model_answer(
            user_answer_en, model_ans, job_title, job_desc, resume_text
        )
        # Save model answer so we don't regenerate it later
        try:
            sess["history"][-1]["model_answer"] = model_ans
        except Exception:
            pass
    except Exception:
        return {"ai_feedback": ""}
    return {"ai_feedback": feedback}




def generate_model_answer_web(session_id: str) -> Dict[str, Any]:
    sess = WEB_SESSIONS.get(session_id)
    if not sess:
        return {"model_answer": ""}
    history = sess.get("history", [])
    question = (history[-1]["question"] if history else sess.get("last_question") or "")
    job_title = sess.get("job_title","")
    job_desc  = sess.get("job_description") or sess.get("job_desc") or ""
    resume_text = sess.get("resume_text") or sess.get("resume_input") or ""
    # Try to reuse a previously stored model answer
    if history:
        stored_answer = history[-1].get("model_answer")
        if stored_answer:
            return {"model_answer": stored_answer}
    # Otherwise, generate a new one
    try:
        answer = generate_reference_answer(question, job_title, job_desc, resume_text)
    except Exception:
        answer = ""
    return {"model_answer": answer}


def start_recorded_interview_web(job_title: str,
                                 job_description: str,
                                 interview_type: str,
                                 resume_input: str,
                                 experience_level: str,
                                 manual_type: bool,
                                 language: str) -> Dict[str, Any]:
    """
    Start a recorded interview session.  This function wraps
    `start_interview_web` but also records the candidate’s chosen language
    (`language`) so that later answers are transcribed without running
    automatic language detection.
    """
    payload = start_interview_web(job_title, job_description, interview_type,
                                  resume_input, experience_level, manual_type)
    sid = payload.get("session_id")
    sess = WEB_SESSIONS.get(sid)
    if sess is not None:
        sess["override_language"] = language
        sess["language"] = language
        sess["mismatch_count"] = 0
    return payload


def process_recorded_answer_web(session_id: str, audio_path: str) -> Dict[str, Any]:
    """
    Process an answer in recorded interview mode.  This thin wrapper simply
    calls `process_answer_web` with the session’s stored language.
    """
    sess = WEB_SESSIONS.get(session_id)
    if not sess:
        raise ValueError("Invalid session id")
    lang = sess.get("language") or sess.get("override_language") or "en"
    return process_answer_web(session_id, audio_path, manual_language=lang)


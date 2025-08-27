# app.py — UPDATED

import os, uuid, json, traceback
from typing import Any, Dict
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

# === Paths ===
BASE_DIR   = os.path.abspath(os.path.dirname(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MEDIA_DIR  = os.path.join(DATA_DIR, "media")
RESUME_DIR = os.path.join(DATA_DIR, "resumes")
os.makedirs(MEDIA_DIR, exist_ok=True)
os.makedirs(RESUME_DIR, exist_ok=True)

# === Flask ===
app = Flask(__name__)

# === In-memory session book-keeping for the web wrapper ===
SESSIONS: Dict[str, Dict[str, Any]] = {}  # session_id -> { job_title, ..., last_original_path, last_cleaned_path }

# === Import your pipeline (same directory) ===
import pipeline as pl

# If you expose these from pipeline, we’ll use them directly.
DROPDOWN_INTERVIEW_TYPES = getattr(pl, "DROPDOWN_INTERVIEW_TYPES", {
    "1": "Behavioral Interview",
    "2": "Technical Interview",
    "3": "Situational Interview",
    "4": "Competency-Based Interview",
    "5": "Ethical or Integrity-Based Interview",
})

# ---------------------- helpers ----------------------
def save_upload(f, folder) -> str:
    name = secure_filename(f.filename or f"file-{uuid.uuid4().hex}")
    path = os.path.join(folder, name)
    f.save(path)
    return path

try:
    from pydub import AudioSegment
except Exception:
    AudioSegment = None

def convert_to_wav_if_needed(input_path: str) -> str:
    """
    Convert to .wav if file extension suggests non-wav and pydub is available.
    """
    lower = input_path.lower()
    needs_wav = lower.endswith((".webm", ".m4a", ".mp3", ".ogg"))
    if not needs_wav or not AudioSegment:
        return input_path
    try:
        sound = AudioSegment.from_file(input_path)
        wav_path = os.path.splitext(input_path)[0] + ".wav"
        sound.export(wav_path, format="wav")
        return wav_path
    except Exception:
        # Fall back to original if conversion fails
        return input_path

def media_url_for(path: str) -> str:
    """
    Turn an absolute file path in DATA_DIR into a URL served from /media.
    """
    rel = os.path.relpath(path, DATA_DIR).replace("\\", "/")
    return f"/media/{rel}"

# ---- normalize any transcript-like object to a string ----
def _to_text(val):
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        for k in ("text", "transcript", "content"):
            if k in val and isinstance(val[k], str):
                return val[k]
        try:
            return json.dumps(val, ensure_ascii=False)
        except Exception:
            return str(val)
    if isinstance(val, list):
        parts = []
        for item in val:
            s = _to_text(item)
            if s:
                parts.append(s)
        return " ".join(parts)
    return "" if val is None else str(val)

# ---------------------- pipeline adapters ----------------------
def pipeline_start_interview(job_title, job_description, interview_type,
                             resume_input, experience_level, manual_type) -> Dict[str, Any]:
    """
    Prefer web-aware wrapper if present.
    """
    if hasattr(pl, "start_interview_web"):
        return pl.start_interview_web(job_title, job_description, interview_type,
                                      resume_input, experience_level, manual_type)
    # Fallback to CLI signature
    q = pl.start_interview(job_title, job_description, interview_type,
                           resume_input, experience_level, manual_type)
    return {"session_id": uuid.uuid4().hex, "question": str(q)}

def pipeline_process_answer(session_id: str, audio_path: str) -> Dict[str, Any]:
    """
    Call web-aware answer processing and normalize to UI shape.
    """
    if hasattr(pl, "process_answer_web"):
        out = pl.process_answer_web(session_id, audio_path)
    else:
        out = pl.process_answer(session_id, audio_path)

    # Keep the cleaned path in our session so /voice_analysis can run later
    cleaned_path = out.get("cleaned_audio_path", "")
    if cleaned_path:
        SESSIONS.setdefault(session_id, {})["last_cleaned_path"] = cleaned_path

    # If pipeline saved/updated history, also mirror a file locally (defensive)
    if hasattr(pl, "save_history_json_web"):
        try:
            pl.save_history_json_web(session_id)
        except Exception:
            pass

    # --- NEW: normalize transcripts and save to session ---
    native = _to_text(out.get("native_text", ""))
    english = _to_text(out.get("english_translation", ""))

    SESSIONS.setdefault(session_id, {})["last_transcript"] = {
        "native": native,
        "english": english,
        "language": out.get("language", "")
    }

    # --- replaced result block to ensure strings + save transcript ---
    result = {
        "mismatch": bool(out.get("mismatch", False)),
        "mismatchMessage": str(out.get("mismatchMessage", "")),
        "manualLanguageNeeded": bool(out.get("manualLanguageNeeded", False)),
        "langs": out.get("langs", []),
        "originalAudioURL": "",
        "cleanedAudioURL": media_url_for(out.get("cleaned_audio_path", "")) if out.get("cleaned_audio_path") else "",
        "nativeText": native,
        "englishText": english,
        "detectedLang": out.get("language", ""),
        "aiFeedback": "",   # placeholder; generated on demand
        "modelAnswer": ""   # placeholder; generated on demand
    }

    return result



def pipeline_next_question(session_id: str) -> str:
    if hasattr(pl, "next_question_web"):
        return pl.next_question_web(session_id)
    if hasattr(pl, "next_question"):
        return pl.next_question(session_id)
    if hasattr(pl, "get_next_question"):
        return pl.get_next_question(session_id)
    return "Describe a recent challenge you faced and how you handled it."

# ---------------------- routes ----------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/start_interview", methods=["POST"])
def start_interview_route():
    """
    Form-data:
      jobTitle, jobDesc, resumeChoice('yes'|'no'), resumeFile(PDF),
      experienceLevel('1'|'2'|'3'),
      typeMethod('manual'|'list'), manualType OR dropdownType
    """
    try:
        form = request.form
        files = request.files

        job_title = form.get("jobTitle", "").strip()
        job_desc  = form.get("jobDesc", "").strip()
        exp       = form.get("experienceLevel", "").strip()
        type_meth = form.get("typeMethod", "manual")
        manual    = (type_meth == "manual")
        if manual:
            interview_type = form.get("manualType", "").strip()
        else:
            idx = form.get("dropdownType", "").strip()
            interview_type = DROPDOWN_INTERVIEW_TYPES.get(idx, idx)

        resume_choice = form.get("resumeChoice", "no")
        resume_input  = ""
        if resume_choice == "yes":
            f = files.get("resumeFile")
            if f and f.filename:
                saved = save_upload(f, RESUME_DIR)
                # You said your pipeline expects a path; pass the saved path.
                resume_input = saved  # keep path; your pipeline can open and parse.

        payload = pipeline_start_interview(job_title, job_desc, interview_type,
                                           resume_input, exp, manual)
        session_id = payload.get("session_id")
        question   = payload.get("question", "")

        # Minimal mirror for UI convenience
        SESSIONS[session_id] = {
            "job_title": job_title, "job_desc": job_desc,
            "experience_level": exp, "interview_type": interview_type,
            "manual_type": manual, "resume_path_or_text": resume_input,
            "last_original_path": "", "last_cleaned_path": ""
        }

        return jsonify({"ok": True, "session_id": session_id, "question": question})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/process_answer", methods=["POST"])
def process_answer_route():
    """
    Form-data:
      session_id, audio (recorded blob)
    """
    try:
        session_id = request.form.get("session_id", "").strip()
        if not session_id or session_id not in SESSIONS:
            return jsonify({"ok": False, "error": "Invalid or missing session_id"}), 400

        audio = request.files.get("audio")
        if not audio:
            return jsonify({"ok": False, "error": "Missing audio file"}), 400

        # Save original audio
        filename = secure_filename(audio.filename or f"audio-{uuid.uuid4().hex}.webm")
        original_path = os.path.join(MEDIA_DIR, filename)
        audio.save(original_path)
        SESSIONS[session_id]["last_original_path"] = original_path

        # Convert if needed for the pipeline
        pipeline_input_path = convert_to_wav_if_needed(original_path)

        # Process via pipeline
        out = pipeline_process_answer(session_id, pipeline_input_path)
        out["originalAudioURL"] = media_url_for(original_path)
        return jsonify({"ok": True, **out})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/manual_language", methods=["POST"])
def manual_language_route():
    """
    JSON: { session_id, language }  where language in ['en','ml','kn']
    Completes the step using the last recorded audio without re-recording.
    """
    try:
        data = request.get_json(force=True)
        sid  = data.get("session_id", "")
        lang = data.get("language", "")
        if not sid or sid not in SESSIONS:
            return jsonify({"ok": False, "error": "Invalid session_id"}), 400
        if lang not in {"en", "ml", "kn"}:
            return jsonify({"ok": False, "error": "Invalid language"}), 400

        # Use last audio path from session
        original = SESSIONS[sid].get("last_original_path", "")
        if not original:
            return jsonify({"ok": False, "error": "No audio available to finalize"}), 400

        pipeline_input = convert_to_wav_if_needed(original)

        if hasattr(pl, "finalize_with_manual_language_web"):
            out = pl.finalize_with_manual_language_web(sid, pipeline_input, lang)
        else:
            # Fallback: run standard processing but force language inside pipeline
            out = pl.process_answer_web(sid, pipeline_input, manual_language=lang)

        # Keep cleaned path for later voice analysis
        cleaned_path = out.get("cleaned_audio_path", "")
        if cleaned_path:
            SESSIONS[sid]["last_cleaned_path"] = cleaned_path

        # Save history file if exposed
        if hasattr(pl, "save_history_json_web"):
            try:
                pl.save_history_json_web(sid)
            except Exception:
                pass

        # --- NEW: normalize transcripts and save to session ---
        native = _to_text(out.get("native_text", ""))
        english = _to_text(out.get("english_translation", ""))

        SESSIONS[sid]["last_transcript"] = {
            "native": native,
            "english": english,
            "language": out.get("language", lang)
        }

        # --- replaced result block to ensure strings + save transcript ---
        result = {
            "ok": True,
            "nativeText": native,
            "englishText": english,
            "detectedLang": out.get("language", lang),
            "aiFeedback": out.get("ai_feedback", ""),
            "modelAnswer": out.get("model_answer", ""),
            "cleanedAudioURL": media_url_for(cleaned_path) if cleaned_path else "",
            "originalAudioURL": media_url_for(original)
        }
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/ai_feedback", methods=["POST"])
def ai_feedback_route():
    data = request.get_json(force=True)
    sid = data.get("session_id", "")
    if not sid or sid not in SESSIONS:
        return jsonify({"ok": False, "error": "Invalid session_id"}), 400
    if hasattr(pl, "generate_ai_feedback_web"):
        res = pl.generate_ai_feedback_web(sid)
        feedback = res.get("ai_feedback", "")
    else:
        feedback = ""
    return jsonify({"ok": True, "aiFeedback": feedback})

@app.route("/voice_analysis", methods=["POST"])
def voice_analysis_route():
    try:
        data = request.get_json(force=True)
        sid = data.get("session_id", "")
        if not sid or sid not in SESSIONS:
            return jsonify({"ok": False, "error": "Invalid session_id"}), 400

        cleaned = SESSIONS[sid].get("last_cleaned_path", "")
        target  = cleaned or SESSIONS[sid].get("last_original_path", "")
        if not target:
            return jsonify({"ok": False, "error": "No audio to analyze"}), 400

        transcript_data = SESSIONS[sid].get("last_transcript", {})

        # STRICT: require language + native transcript for analysis
        if not (transcript_data.get("language") and (transcript_data.get("native") or transcript_data.get("native_text"))):
            return jsonify({"ok": False, "error": "Missing language or native transcript for voice analysis"}), 400

        if hasattr(pl, "analyze_audio_web"):
            metrics = pl.analyze_audio_web(sid, target, transcript_data)
        else:
            try:
                metrics = pl.analyze_audio(target, transcript_data)
            except TypeError:
                metrics = pl.analyze_audio(target)

        return jsonify({"ok": True, "voiceAnalysis": metrics})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500



@app.route("/model_answer", methods=["POST"])
def model_answer_route():
    data = request.get_json(force=True)
    sid = data.get("session_id", "")
    if not sid or sid not in SESSIONS:
        return jsonify({"ok": False, "error": "Invalid session_id"}), 400
    if hasattr(pl, "generate_model_answer_web"):
        res = pl.generate_model_answer_web(sid)
        answer = res.get("model_answer", "")
    else:
        answer = ""
    return jsonify({"ok": True, "modelAnswer": answer})

@app.route("/next_question", methods=["POST"])
def next_question_route():
    try:
        sid = request.form.get("session_id") or (request.json or {}).get("session_id")
        if not sid or sid not in SESSIONS:
            return jsonify({"ok": False, "error": "Invalid or missing session_id"}), 400
        q = pipeline_next_question(sid)
        return jsonify({"ok": True, "question": q})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500

# Serve any file under data/ via /media
@app.route("/media/<path:relpath>", methods=["GET"])
def media(relpath: str):
    return send_from_directory(DATA_DIR, relpath, as_attachment=False)


@app.route("/start_recorded_interview", methods=["POST"])
def start_recorded_interview_route():
    """
    Start a new recorded interview session.  This endpoint mirrors
    /start_interview but requires a language choice.
    """
    try:
        form = request.form
        files = request.files

        job_title = form.get("jobTitle", "").strip()
        job_desc  = form.get("jobDesc", "").strip()
        exp       = form.get("experienceLevel", "").strip()
        type_meth = form.get("typeMethod", "manual")
        manual    = (type_meth == "manual")
        if manual:
            interview_type = form.get("manualType", "").strip()
        else:
            idx = form.get("dropdownType", "").strip()
            interview_type = DROPDOWN_INTERVIEW_TYPES.get(idx, idx)

        resume_choice = form.get("resumeChoice", "no")
        resume_input  = ""
        if resume_choice == "yes":
            f = files.get("resumeFile")
            if f and f.filename:
                saved = save_upload(f, RESUME_DIR)
                resume_input = saved

        language = form.get("languageChoice", "").strip().lower()
        if language not in {"en", "ml", "kn"}:
            return jsonify({"ok": False, "error": "Invalid or missing language choice"}), 400

        payload = pl.start_recorded_interview_web(job_title, job_desc, interview_type,
                                                  resume_input, exp, manual, language)
        session_id = payload.get("session_id")
        question   = payload.get("question", "")

        SESSIONS[session_id] = {
            "job_title": job_title, "job_desc": job_desc,
            "experience_level": exp, "interview_type": interview_type,
            "manual_type": manual, "resume_path_or_text": resume_input,
            "last_original_path": "", "last_cleaned_path": "",
            "language": language
        }

        return jsonify({"ok": True, "session_id": session_id, "question": question})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/process_recorded_answer", methods=["POST"])
def process_recorded_answer_route():
    """
    Process an uploaded recording in recorded mode.  Requires session_id and audio.
    """
    try:
        session_id = request.form.get("session_id", "").strip()
        if not session_id or session_id not in SESSIONS:
            return jsonify({"ok": False, "error": "Invalid or missing session_id"}), 400

        audio = request.files.get("audio")
        if not audio:
            return jsonify({"ok": False, "error": "Missing audio file"}), 400

        filename = secure_filename(audio.filename or f"audio-{uuid.uuid4().hex}.webm")
        original_path = os.path.join(MEDIA_DIR, filename)
        audio.save(original_path)
        SESSIONS[session_id]["last_original_path"] = original_path

        pipeline_input_path = convert_to_wav_if_needed(original_path)
        out = pl.process_recorded_answer_web(session_id, pipeline_input_path)

        cleaned_path = out.get("cleaned_audio_path", "")
        if cleaned_path:
            SESSIONS[session_id]["last_cleaned_path"] = cleaned_path

        native = _to_text(out.get("native_text", ""))
        english = _to_text(out.get("english_translation", ""))
        SESSIONS[session_id]["last_transcript"] = {
            "native": native,
            "english": english,
            "language": out.get("language", SESSIONS[session_id].get("language", ""))
        }

        result = {
            "ok": True,
            "nativeText": native,
            "englishText": english,
            "detectedLang": out.get("language", SESSIONS[session_id].get("language", "")),
            "cleanedAudioURL": media_url_for(cleaned_path) if cleaned_path else "",
            "originalAudioURL": media_url_for(original_path)
        }
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    # Heavy audio libs + Windows reloader can clash; keep reloader off.
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False, threaded=True)

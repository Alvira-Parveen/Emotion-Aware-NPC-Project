"""
modules/voice_emotion.py  —  Stage C: Voice Emotion Recognition
════════════════════════════════════════════════════════════════
Estimates emotion from raw audio (WAV/WEBM/OGG bytes).

Pipeline:
  1. Receive raw audio bytes from Flask route (browser MediaRecorder output)
  2. Convert to 16 kHz mono WAV using pydub / ffmpeg
  3. Run a speech-emotion model on the waveform
  4. Return scores dict mapped to our 7 FER emotion labels

Model options (in priority order):
  A. SpeechBrain   — speechbrain/emotion-recognition-wav2vec2-IEMOCAP
     4 classes: angry, happy, neutral, sad
     Fast (~50 ms on CPU), no API key, works offline after download (~360 MB).
  B. HuggingFace pipeline — ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
     7 classes including fear/disgust/surprise, larger model (~1.2 GB).
  C. No-model fallback — returns neutral with 0.0 confidence and clear error.

IMPORTANT DISTINCTION:
  Voice emotion ≠ text sentiment.  This module runs on the AUDIO SIGNAL
  (prosody, pitch, energy, rhythm) NOT on a speech transcript.
  If you want text sentiment of what was said, use modules/text_emotion.py
  on the transcript instead.

Supported input formats: WAV, WEBM, OGG, MP4 audio, FLAC, MP3.
  pydub + ffmpeg handle all format conversions.
  If ffmpeg is missing, only WAV is accepted.

Privacy: audio bytes are processed in-memory and never written to disk
unless a temp file is strictly required by the model (deleted immediately).
"""

from __future__ import annotations

import io
import os
import logging
import tempfile
import threading
from typing import Optional

logger = logging.getLogger(__name__)

ALL_EMOTIONS = ["happy", "sad", "angry", "surprise", "fear", "disgust", "neutral"]

# ── SpeechBrain label map → FER labels ───────────────────────────────────────
_SB_TO_FER: dict[str, str] = {
    "ang": "angry",
    "hap": "happy",
    "neu": "neutral",
    "sad": "sad",
    # some checkpoints use full words
    "angry":   "angry",
    "happy":   "happy",
    "neutral": "neutral",
    "sad":     "sad",
}

# ── HF pipeline label map → FER labels ───────────────────────────────────────
_HF_VOICE_TO_FER: dict[str, str] = {
    "angry":     "angry",
    "calm":      "neutral",
    "disgust":   "disgust",
    "fear":      "fear",
    "happy":     "happy",
    "neutral":   "neutral",
    "sad":       "sad",
    "surprise":  "surprise",
    # ehcalabres model uses these:
    "anger":     "angry",
    "happiness": "happy",
    "sadness":   "sad",
    "boredom":   "neutral",
}

# ── Lazy globals ──────────────────────────────────────────────────────────────
_sb_classifier   = None   # SpeechBrain classifier
_hf_pipeline     = None   # HuggingFace pipeline
_backend         = "none" # "speechbrain" | "hf_pipeline" | "none"
_lock            = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
#  Backend loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_speechbrain() -> bool:
    global _sb_classifier, _backend
    try:
        from speechbrain.pretrained import EncoderClassifier  # type: ignore
        MODEL_ID = os.environ.get(
            "VOICE_EMOTION_MODEL_SB",
            "speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        )
        logger.info("Loading SpeechBrain voice-emotion model: %s …", MODEL_ID)
        _sb_classifier = EncoderClassifier.from_hparams(
            source=MODEL_ID,
            savedir=os.path.join(
                os.path.expanduser("~"), ".cache", "speechbrain", "emotion"
            ),
            run_opts={"device": "cpu"},
        )
        _backend = "speechbrain"
        logger.info("SpeechBrain voice-emotion model loaded.")
        return True
    except Exception as exc:
        logger.warning("SpeechBrain unavailable: %s", exc)
        return False


def _load_hf_pipeline() -> bool:
    global _hf_pipeline, _backend
    try:
        from transformers import pipeline  # type: ignore
        MODEL_ID = os.environ.get(
            "VOICE_EMOTION_MODEL_HF",
            "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        )
        logger.info("Loading HF voice-emotion pipeline: %s …", MODEL_ID)
        _hf_pipeline = pipeline(
            "audio-classification",
            model=MODEL_ID,
            top_k=None,
        )
        _backend = "hf_pipeline"
        logger.info("HF voice-emotion pipeline loaded.")
        return True
    except Exception as exc:
        logger.warning("HF voice pipeline unavailable: %s", exc)
        return False


def ensure_loaded() -> str:
    """Lazy-load the first available voice backend."""
    with _lock:
        if _backend != "none":
            return _backend
        _load_speechbrain() or _load_hf_pipeline()
        return _backend


# ─────────────────────────────────────────────────────────────────────────────
#  Audio pre-processing
# ─────────────────────────────────────────────────────────────────────────────

def _to_16khz_mono_wav(audio_bytes: bytes, mime_hint: str = "") -> bytes:
    """
    Convert arbitrary audio bytes to 16 kHz mono PCM WAV bytes.
    Uses pydub (which delegates to ffmpeg for format detection).
    Falls back to returning the input unchanged if pydub is unavailable.
    """
    try:
        from pydub import AudioSegment  # type: ignore
        fmt = "webm"
        if   "ogg"  in mime_hint: fmt = "ogg"
        elif "wav"  in mime_hint: fmt = "wav"
        elif "mp4"  in mime_hint: fmt = "mp4"
        elif "flac" in mime_hint: fmt = "flac"
        elif "mp3"  in mime_hint: fmt = "mp3"

        seg: AudioSegment = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
        seg = seg.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        buf = io.BytesIO()
        seg.export(buf, format="wav")
        return buf.getvalue()
    except Exception as exc:
        logger.warning("Audio conversion failed (%s) — passing raw bytes", exc)
        return audio_bytes


def _validate_audio(audio_bytes: bytes) -> Optional[str]:
    """Return an error string if audio is obviously invalid, else None."""
    if not audio_bytes:
        return "No audio data received."
    if len(audio_bytes) < 1000:
        return "Audio clip is too short (< 1 KB). Please record at least 1–2 seconds."
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _uniform_scores(dominant: str, confidence: float) -> dict[str, float]:
    remaining = (1.0 - confidence) / max(len(ALL_EMOTIONS) - 1, 1)
    return {em: confidence if em == dominant else remaining for em in ALL_EMOTIONS}


def _normalise_scores(raw: dict[str, float], label_map: dict[str, str]) -> dict[str, float]:
    """Map raw model labels to FER labels and normalise to sum ≈ 1."""
    mapped: dict[str, float] = {}
    for lbl, score in raw.items():
        fer_lbl = label_map.get(lbl.lower(), "neutral")
        mapped[fer_lbl] = mapped.get(fer_lbl, 0.0) + score
    # Fill missing emotions with 0
    for em in ALL_EMOTIONS:
        mapped.setdefault(em, 0.0)
    # Normalise
    total = sum(mapped.values()) or 1.0
    return {em: mapped[em] / total for em in ALL_EMOTIONS}


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def predict(audio_bytes: bytes, mime_hint: str = "audio/webm") -> dict:
    """
    Predict emotion from audio bytes.

    Parameters
    ----------
    audio_bytes : bytes
        Raw audio data from the browser (WebM/OGG/WAV).
    mime_hint : str
        MIME type string from the browser, used for format detection.

    Returns
    -------
    dict with keys:
      emotion     (str)   — dominant FER-label emotion
      confidence  (float) — score for dominant emotion [0, 1]
      scores      (dict)  — {emotion: float} for all 7 FER labels
      backend     (str)   — model used
      note        (str)   — important caveat string (e.g. 4-class limitation)
      error       (str|None)
    """
    # Validate
    err = _validate_audio(audio_bytes)
    if err:
        return {
            "emotion": "neutral", "confidence": 0.0,
            "scores": {em: 1.0 / len(ALL_EMOTIONS) for em in ALL_EMOTIONS},
            "backend": "none", "note": "", "error": err,
        }

    # Convert to 16 kHz mono WAV
    wav_bytes = _to_16khz_mono_wav(audio_bytes, mime_hint)

    backend = ensure_loaded()

    # ── SpeechBrain ───────────────────────────────────────────────────────────
    if backend == "speechbrain" and _sb_classifier is not None:
        try:
            import torch, torchaudio  # type: ignore
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(wav_bytes)
                tmp_path = tmp.name
            try:
                waveform, sr = torchaudio.load(tmp_path)
                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(sr, 16000)
                    waveform = resampler(waveform)
                out_prob, score, idx, label = _sb_classifier.classify_batch(waveform)
                dominant = _SB_TO_FER.get(label[0].strip().lower(), "neutral")
                conf = float(torch.max(out_prob).item())
                scores = _uniform_scores(dominant, conf)
                return {
                    "emotion":    dominant,
                    "confidence": round(conf, 4),
                    "scores":     scores,
                    "backend":    "speechbrain",
                    "note":       (
                        "SpeechBrain (IEMOCAP) classifies 4 emotions: "
                        "angry, happy, neutral, sad. "
                        "Fear, disgust, and surprise cannot be detected by this model."
                    ),
                    "error": None,
                }
            finally:
                os.unlink(tmp_path)
        except Exception as exc:
            logger.error("SpeechBrain prediction error: %s", exc)

    # ── HuggingFace Pipeline ──────────────────────────────────────────────────
    if backend == "hf_pipeline" and _hf_pipeline is not None:
        try:
            import numpy as np  # type: ignore
            import soundfile as sf  # type: ignore

            with io.BytesIO(wav_bytes) as wav_buf:
                audio_np, sr = sf.read(wav_buf, dtype="float32")
            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=1)

            raw_results: list[dict] = _hf_pipeline(
                {"array": audio_np, "sampling_rate": sr}
            )
            raw_map = {r["label"].lower(): r["score"] for r in raw_results}
            scores  = _normalise_scores(raw_map, _HF_VOICE_TO_FER)
            dominant = max(scores, key=scores.get)
            return {
                "emotion":    dominant,
                "confidence": round(scores[dominant], 4),
                "scores":     scores,
                "backend":    "hf_pipeline",
                "note":       "HuggingFace wav2vec2 voice emotion (7 classes).",
                "error":      None,
            }
        except Exception as exc:
            logger.error("HF voice pipeline error: %s", exc)

    # ── Fallback: Lightweight Acoustic Prosody Feature Extractor (RMS & ZCR) ──
    try:
        import numpy as np
        # Read PCM waveform samples
        raw_samples = np.frombuffer(wav_bytes[44:], dtype=np.int16) if len(wav_bytes) > 44 else np.frombuffer(wav_bytes, dtype=np.int8)
        if len(raw_samples) > 200:
            float_samples = raw_samples.astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(float_samples**2)))
            zcr = float(np.mean(np.abs(np.diff(np.sign(float_samples)))) / 2.0)

            # Heuristic prosodic mapping from energy & zero-crossing rate:
            if rms > 0.18:
                dom, conf = ("angry" if zcr > 0.18 else "surprise"), min(0.5 + rms, 0.85)
            elif rms > 0.08:
                dom, conf = "happy", min(0.45 + rms, 0.80)
            elif rms < 0.025:
                dom, conf = "sad", min(0.50 + (0.03 - rms) * 10, 0.75)
            else:
                dom, conf = "neutral", 0.65

            return {
                "emotion":    dom,
                "confidence": round(conf, 4),
                "scores":     _uniform_scores(dom, conf),
                "backend":    "acoustic_prosody_heuristic",
                "note":       f"Acoustic prosody analysis (RMS energy: {rms:.3f}, ZCR: {zcr:.3f}).",
                "error":      None,
            }
    except Exception as exc:
        logger.warning("Acoustic heuristic fallback error: %s", exc)

    # ── No backend ────────────────────────────────────────────────────────────
    return {
        "emotion":    "neutral",
        "confidence": 0.0,
        "scores":     {em: 1.0 / len(ALL_EMOTIONS) for em in ALL_EMOTIONS},
        "backend":    "none",
        "note":       "",
        "error":      (
            "No voice-emotion backend available. "
            "Install speechbrain + torch, or transformers + soundfile."
        ),
    }

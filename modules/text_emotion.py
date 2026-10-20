"""
modules/text_emotion.py  —  Stage B: Text Emotion Recognition
═══════════════════════════════════════════════════════════════
Uses the HuggingFace pipeline with:
  Primary model : j-hartmann/emotion-english-distilroberta-base
                  (7 Ekman emotions — anger, disgust, fear, joy,
                   neutral, sadness, surprise)
  Fallback      : VADER lexicon-based sentiment (positive/negative/neutral)
                  when transformers or the model are unavailable.

This module is intentionally self-contained and has no dependency on the
face or voice emotion modules so it can be imported, tested, and replaced
independently.

Label mapping to the 7 FER labels used by web_app.py:
  joy      → happy
  anger    → angry
  sadness  → sad
  surprise → surprise
  fear     → fear
  disgust  → disgust
  neutral  → neutral
"""

from __future__ import annotations

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Canonical label map (HF model → FER labels) ──────────────────────────────
_HF_TO_FER: dict[str, str] = {
    "joy":      "happy",
    "anger":    "angry",
    "sadness":  "sad",
    "surprise": "surprise",
    "fear":     "fear",
    "disgust":  "disgust",
    "neutral":  "neutral",
}

# All 7 emotion labels used across the system
ALL_EMOTIONS = ["happy", "sad", "angry", "surprise", "fear", "disgust", "neutral"]

# ── Lazy globals ──────────────────────────────────────────────────────────────
_pipeline = None           # transformers pipeline instance
_vader    = None           # VADER SentimentIntensityAnalyzer instance
_backend  = "none"         # which backend is active: "transformers" | "vader" | "none"
_loading  = False


def _load_transformers_pipeline() -> bool:
    """
    Attempt to load the HuggingFace transformers pipeline.
    Returns True on success, False if transformers is unavailable.
    The model is downloaded to ~/.cache/huggingface on first use (~67 MB).
    """
    global _pipeline, _backend
    try:
        from transformers import pipeline  # type: ignore
        MODEL_ID = os.environ.get(
            "TEXT_EMOTION_MODEL",
            "j-hartmann/emotion-english-distilroberta-base",
        )
        logger.info("Loading text-emotion model: %s …", MODEL_ID)
        _pipeline = pipeline(
            "text-classification",
            model=MODEL_ID,
            top_k=None,           # return ALL label scores, not just top-1
            truncation=True,
            max_length=512,
        )
        _backend = "transformers"
        logger.info("Text-emotion model loaded (backend: transformers).")
        return True
    except Exception as exc:
        logger.warning("transformers backend unavailable: %s", exc)
        return False


def _load_vader_fallback() -> bool:
    """
    Attempt to load VADER as a lightweight fallback.
    VADER only returns positive/negative/neutral sentiment — not fine-grained
    emotions.  We map it to the three most likely FER labels for transparency.
    Returns True on success.
    """
    global _vader, _backend
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
        _vader   = SentimentIntensityAnalyzer()
        _backend = "vader"
        logger.info("Text-emotion backend: VADER (coarse sentiment only).")
        return True
    except Exception as exc:
        logger.warning("VADER fallback unavailable: %s", exc)
        return False


def ensure_loaded() -> str:
    """
    Ensure a backend is loaded.  Called lazily on first predict().
    Returns the active backend name: "transformers" | "vader" | "none".
    """
    global _loading
    if _backend != "none":
        return _backend
    if _loading:
        return "loading"
    _loading = True
    _load_transformers_pipeline() or _load_vader_fallback()
    _loading = False
    return _backend


def _uniform_scores(dominant: str, confidence: float) -> dict[str, float]:
    """
    Build a full 7-emotion scores dict when only dominant + confidence are known.
    Remaining mass is spread uniformly over the other six emotions.
    """
    remaining = (1.0 - confidence) / (len(ALL_EMOTIONS) - 1)
    return {
        em: confidence if em == dominant else remaining
        for em in ALL_EMOTIONS
    }


def predict(text: str) -> dict:
    """
    Estimate emotion from a text string.

    Returns a dict with:
      emotion     (str)   — dominant FER-label emotion
      confidence  (float) — raw score for dominant emotion [0, 1]
      scores      (dict)  — {emotion: float} for all 7 labels
      backend     (str)   — which model was used
      error       (str|None) — error message if prediction failed

    Empty or whitespace-only input is handled gracefully.
    """
    text = (text or "").strip()
    if not text:
        return {
            "emotion":    "neutral",
            "confidence": 1.0,
            "scores":     {em: (1.0 if em == "neutral" else 0.0) for em in ALL_EMOTIONS},
            "backend":    "rule",
            "error":      "Empty input — defaulting to neutral.",
        }

    backend = ensure_loaded()

    # ── Transformers (primary) ────────────────────────────────────────────────
    if backend == "transformers" and _pipeline is not None:
        try:
            raw: list[dict] = _pipeline(text)[0]
            # raw = [{"label": "joy", "score": 0.92}, …]
            fer_scores: dict[str, float] = {}
            for item in raw:
                fer_label = _HF_TO_FER.get(item["label"].lower(), item["label"].lower())
                # Accumulate in case of label collisions (shouldn't happen here)
                fer_scores[fer_label] = fer_scores.get(fer_label, 0.0) + item["score"]

            # Ensure all 7 labels are present
            for em in ALL_EMOTIONS:
                fer_scores.setdefault(em, 0.0)

            dominant = max(fer_scores, key=fer_scores.get)
            return {
                "emotion":    dominant,
                "confidence": fer_scores[dominant],
                "scores":     fer_scores,
                "backend":    "transformers",
                "error":      None,
            }
        except Exception as exc:
            logger.error("Transformers prediction error: %s", exc)

    # ── VADER + Lexicon (fallback) ───────────────────────────────────────────
    if backend == "vader" and _vader is not None:
        try:
            # Check fine-grained emotion lexicon cues first
            lower_text = text.lower()
            words = set(lower_text.replace("!", " ").replace("?", " ").replace(".", " ").replace(",", " ").split())
            
            lexicon_cues = {
                "happy":    {"excited", "happy", "joy", "glad", "delighted", "awesome", "great", "wonderful", "fantastic", "cheerful", "thrilled", "love", "smile", "ecstatic"},
                "angry":    {"angry", "mad", "furious", "annoyed", "rage", "irritated", "pissed", "hate", "outraged"},
                "sad":      {"sad", "unhappy", "depressed", "sorrow", "grief", "gloomy", "miserable", "heartbroken", "down", "cry", "crying"},
                "surprise": {"surprise", "surprised", "shocked", "astonished", "amazed", "wow", "omg", "unbelievable"},
                "fear":     {"fear", "afraid", "scared", "terrified", "frightened", "panicked", "nervous", "anxious", "horror"},
                "disgust":  {"disgust", "disgusted", "gross", "nasty", "revolting", "repulsed", "yuck", "ew"},
            }
            
            matched_cue = None
            for emo, cue_set in lexicon_cues.items():
                if words & cue_set:
                    matched_cue = emo
                    break

            vs = _vader.polarity_scores(text)
            compound = vs["compound"]

            if matched_cue:
                dominant = matched_cue
                conf = 0.85
            elif compound >= 0.05:
                dominant, conf = "happy", min(0.55 + compound * 0.45, 0.98)
            elif compound <= -0.05:
                # Differentiate anger vs sadness if negative
                dominant, conf = "sad", min(0.55 + abs(compound) * 0.45, 0.98)
            else:
                dominant, conf = "neutral", 0.65

            return {
                "emotion":    dominant,
                "confidence": round(conf, 4),
                "scores":     _uniform_scores(dominant, conf),
                "backend":    "vader",
                "error":      None if matched_cue else "⚠️ VADER fallback (coarse sentiment only — not fine-grained emotion).",
            }
        except Exception as exc:
            logger.error("VADER prediction error: %s", exc)

    # ── No backend available ──────────────────────────────────────────────────
    return {
        "emotion":    "neutral",
        "confidence": 0.0,
        "scores":     {em: 1.0 / len(ALL_EMOTIONS) for em in ALL_EMOTIONS},
        "backend":    "none",
        "error":      "No text-emotion backend available. Install transformers or vaderSentiment.",
    }

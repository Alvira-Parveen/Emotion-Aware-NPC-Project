"""
modules/fusion.py  —  Stage D: Multimodal Emotion Fusion
══════════════════════════════════════════════════════════
Combines face, text, and voice emotion predictions into a single
fused estimate using a configurable weighted average at the
probability (score) level.

Design decisions
────────────────
1.  DECISION-LEVEL vs PROBABILITY-LEVEL FUSION
    We use probability-level (score-level) fusion: each modality
    contributes a 7-D score vector over the shared emotion label set.
    These are normalised and combined by weighted sum.

    This is deliberately simple and fully explainable — the contribution
    of each modality to the final result is visible and configurable.

2.  RAW SCORES ARE NOT CALIBRATED PROBABILITIES
    FER softmax outputs, HF emotion scores, and VADER heuristics are NOT
    calibrated Bayesian probabilities.  They express relative ordering
    within a modality but cannot be meaningfully compared across modalities
    without calibration.  We therefore:
      - Normalise each modality's scores to sum to 1 before weighting.
      - Use configurable weights that can be set per-deployment via env vars.
      - Expose the per-modality scores alongside the fused result so users
        can see the raw information and judge for themselves.

3.  MISSING MODALITY HANDLING
    If a modality is absent (None) or has a low-confidence flag
    (confidence < CONFIDENCE_THRESHOLD), its weight is set to 0 and the
    other modalities' weights are renormalised.

4.  EQUAL-WEIGHT FALLBACK
    If all modalities are missing, we return a uniform distribution and
    report a clear error.

Configuration (environment variables)
─────────────────────────────────────
  FUSION_WEIGHT_FACE   (default 0.5)
  FUSION_WEIGHT_TEXT   (default 0.3)
  FUSION_WEIGHT_VOICE  (default 0.2)
  FUSION_CONFIDENCE_THRESHOLD  (default 0.05)
      Modalities with max-score below this are treated as missing.
"""

from __future__ import annotations

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

ALL_EMOTIONS = ["happy", "sad", "angry", "surprise", "fear", "disgust", "neutral"]

# ── Default weights (can be overridden via environment variables) ─────────────
_DEFAULT_WEIGHTS = {
    "face":  float(os.environ.get("FUSION_WEIGHT_FACE",  0.5)),
    "text":  float(os.environ.get("FUSION_WEIGHT_TEXT",  0.3)),
    "voice": float(os.environ.get("FUSION_WEIGHT_VOICE", 0.2)),
}

_CONFIDENCE_THRESHOLD = float(os.environ.get("FUSION_CONFIDENCE_THRESHOLD", 0.05))


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalise(scores: dict[str, float]) -> dict[str, float]:
    """
    Normalise a score dict so values sum to 1.
    Ensures all 7 emotion labels are present (fills missing with 0.0).
    """
    full = {em: scores.get(em, 0.0) for em in ALL_EMOTIONS}
    total = sum(full.values())
    if total <= 0:
        n = len(ALL_EMOTIONS)
        return {em: 1.0 / n for em in ALL_EMOTIONS}
    return {em: full[em] / total for em in ALL_EMOTIONS}


def _is_valid(modality_result: Optional[dict]) -> bool:
    """
    Return True if a modality result is present and has enough signal
    (max score above threshold; no backend="none" with confidence=0).
    """
    if modality_result is None:
        return False
    if modality_result.get("error") and modality_result.get("confidence", 0) == 0:
        return False
    scores = modality_result.get("scores", {})
    if not scores:
        return False
    if max(scores.values()) < _CONFIDENCE_THRESHOLD:
        return False
    return True


def _extract_scores(result: dict) -> dict[str, float]:
    """Extract and normalise scores from a modality result dict."""
    return _normalise(result.get("scores", {}))


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def fuse(
    face_result:  Optional[dict] = None,
    text_result:  Optional[dict] = None,
    voice_result: Optional[dict] = None,
    weights:      Optional[dict[str, float]] = None,
) -> dict:
    """
    Fuse emotion predictions from one or more modalities.

    Parameters
    ----------
    face_result  : dict or None — output from web_app.py /detect (FER)
    text_result  : dict or None — output from modules/text_emotion.predict()
    voice_result : dict or None — output from modules/voice_emotion.predict()
    weights      : optional override for {"face": w1, "text": w2, "voice": w3}

    Returns
    -------
    dict with keys:
      emotion     (str)   — fused dominant emotion
      confidence  (float) — fused score for dominant emotion
      scores      (dict)  — fused 7-label score vector
      modalities  (dict)  — {name: {present, weight_used, emotion, confidence, scores}}
      weights_used(dict)  — effective weights after missing-modality adjustment
      method      (str)   — description of fusion method used
      warning     (str)   — any calibration / limitation warnings
      error       (str|None)
    """
    if weights is None:
        weights = _DEFAULT_WEIGHTS.copy()

    # ── Validate which modalities are usable ──────────────────────────────────
    valid = {
        "face":  _is_valid(face_result),
        "text":  _is_valid(text_result),
        "voice": _is_valid(voice_result),
    }

    results_map = {"face": face_result, "text": text_result, "voice": voice_result}

    # ── Build effective weights (zero-out missing modalities) ─────────────────
    eff_w = {k: (weights.get(k, 0.0) if valid[k] else 0.0) for k in valid}
    total_w = sum(eff_w.values())

    if total_w == 0:
        # No valid modalities
        uniform = {em: 1.0 / len(ALL_EMOTIONS) for em in ALL_EMOTIONS}
        return {
            "emotion":      "neutral",
            "confidence":   0.0,
            "scores":       uniform,
            "modalities":   {k: {"present": False} for k in valid},
            "weights_used": eff_w,
            "method":       "no_valid_modalities",
            "warning":      (
                "All modalities are missing or returned zero confidence. "
                "Returning uniform distribution."
            ),
            "error": "No valid modality results to fuse.",
        }

    # Normalise weights
    norm_w = {k: eff_w[k] / total_w for k in eff_w}

    # ── Weighted average of score vectors ─────────────────────────────────────
    fused: dict[str, float] = {em: 0.0 for em in ALL_EMOTIONS}
    modality_details: dict[str, dict] = {}

    for mod_name in ("face", "text", "voice"):
        res  = results_map[mod_name]
        w    = norm_w[mod_name]
        if valid[mod_name] and res is not None:
            norm_scores = _extract_scores(res)
            for em in ALL_EMOTIONS:
                fused[em] += w * norm_scores[em]
            modality_details[mod_name] = {
                "present":    True,
                "weight_used": round(w, 4),
                "emotion":    res.get("emotion", "neutral"),
                "confidence": round(res.get("confidence", 0.0), 4),
                "scores":     {em: round(norm_scores[em], 4) for em in ALL_EMOTIONS},
                "backend":    res.get("backend", "unknown"),
                "note":       res.get("note", ""),
            }
        else:
            modality_details[mod_name] = {
                "present":    False,
                "weight_used": 0.0,
            }

    # Round fused scores
    fused = {em: round(fused[em], 4) for em in ALL_EMOTIONS}
    dominant   = max(fused, key=fused.get)
    confidence = fused[dominant]

    # ── Build warning message ─────────────────────────────────────────────────
    warnings = []
    active_mods = [k for k in valid if valid[k]]
    if len(active_mods) == 1:
        warnings.append(
            f"Only one modality ({active_mods[0]}) contributed. "
            "This is not truly multimodal — add more input channels for better results."
        )
    warnings.append(
        "Raw model scores are not calibrated probabilities. "
        "The fused confidence value indicates relative ordering within "
        "this fusion, not an absolute probability of the true emotion."
    )

    return {
        "emotion":      dominant,
        "confidence":   confidence,
        "scores":       fused,
        "modalities":   modality_details,
        "weights_used": {k: round(norm_w[k], 4) for k in norm_w},
        "method":       "probability_level_weighted_average",
        "warning":      " | ".join(warnings),
        "error":        None,
    }


def get_default_weights() -> dict[str, float]:
    """Return the current default fusion weights (from env vars or hardcoded defaults)."""
    return _DEFAULT_WEIGHTS.copy()

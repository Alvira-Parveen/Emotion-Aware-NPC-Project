"""
modules/npc_adapter.py  —  Stage F: Adaptive NPC & Virtual Environment State
══════════════════════════════════════════════════════════════════════════════
Translates the fused emotional state into dynamic interaction adaptations:
  1. Conversational Style / Persona Tone (passed to LLM / Dialogue selector).
  2. Speech Synthesis Pacing (speech rate, pitch) for the browser voice engine.
  3. Action Recommendations / Interaction choices for the user.
  4. Metaverse / Virtual Environment visual lighting & atmospheric tokens
     (ambient theme colors, glow orbs, mood titles).

This module contains clear, deterministic adaptation rules and provides
stateful session history tracking for emotional transitions.
"""

from __future__ import annotations

from typing import Dict, Any, List

# ── Explicit Adaptation Rules Matrix ──────────────────────────────────────────
ADAPTATION_RULES: Dict[str, Dict[str, Any]] = {
    "happy": {
        "style": "celebratory_enthusiastic",
        "description": "Enthusiastic and uplifting; mirrors positive vitality.",
        "tts_rate": 1.05,
        "tts_pitch": 1.15,
        "environment": {
            "ambient_theme": "emerald_aurora",
            "primary_color": "#48c78e",
            "secondary_color": "#2ecc71",
            "lighting": "bright_warm",
            "atmosphere": "Energetic Sunrise",
            "particles": "golden_sparks",
        },
        "interaction_action": "Share an achievement or explore upbeat quests.",
    },
    "sad": {
        "style": "empathic_comforting",
        "description": "Gentle, non-judgmental, unhurried presence.",
        "tts_rate": 0.88,
        "tts_pitch": 0.95,
        "environment": {
            "ambient_theme": "twilight_sanctuary",
            "primary_color": "#6495ed",
            "secondary_color": "#4facfe",
            "lighting": "soft_diffuse",
            "atmosphere": "Quiet Haven",
            "particles": "gentle_raindrops",
        },
        "interaction_action": "Take your time; share what's on your mind or pause.",
    },
    "angry": {
        "style": "deescalating_grounded",
        "description": "Calm, objective, low-reactivity mediator tone.",
        "tts_rate": 0.90,
        "tts_pitch": 0.92,
        "environment": {
            "ambient_theme": "cooling_ember",
            "primary_color": "#e74c3c",
            "secondary_color": "#f5576c",
            "lighting": "subdued_cool",
            "atmosphere": "Calming Grove",
            "particles": "slow_mist",
        },
        "interaction_action": "Engage in guided breath pacing or express grievances.",
    },
    "surprise": {
        "style": "inquisitive_engaging",
        "description": "Engaged and curious; leans into novelty.",
        "tts_rate": 1.08,
        "tts_pitch": 1.20,
        "environment": {
            "ambient_theme": "electric_starlight",
            "primary_color": "#f1c40f",
            "secondary_color": "#f39c12",
            "lighting": "dynamic_spark",
            "atmosphere": "Wonder Nebula",
            "particles": "cosmic_dust",
        },
        "interaction_action": "Examine the unexpected event together.",
    },
    "fear": {
        "style": "reassuring_protective",
        "description": "Steady, safe, predictable; provides stability.",
        "tts_rate": 0.86,
        "tts_pitch": 1.00,
        "environment": {
            "ambient_theme": "crystal_shield",
            "primary_color": "#9b59b6",
            "secondary_color": "#8e44ad",
            "lighting": "protective_glow",
            "atmosphere": "Sanctuary Citadel",
            "particles": "protective_rings",
        },
        "interaction_action": "Anchor yourself in a safe zone; confirm boundaries.",
    },
    "disgust": {
        "style": "validating_solution_focused",
        "description": "Acknowledges discomfort quickly, offers a path to pivot.",
        "tts_rate": 0.95,
        "tts_pitch": 1.00,
        "environment": {
            "ambient_theme": "purifying_spring",
            "primary_color": "#2ecc71",
            "secondary_color": "#16a085",
            "lighting": "clean_refresh",
            "atmosphere": "Clearing Glade",
            "particles": "floating_leaves",
        },
        "interaction_action": "Pivot away from unpleasant elements to preferred options.",
    },
    "neutral": {
        "style": "attentive_companionable",
        "description": "Balanced, polite, open-ended conversationalist.",
        "tts_rate": 0.98,
        "tts_pitch": 1.00,
        "environment": {
            "ambient_theme": "celestial_orbit",
            "primary_color": "#95a5a6",
            "secondary_color": "#7f8c8d",
            "lighting": "balanced_ambient",
            "atmosphere": "Harmonic Nexus",
            "particles": "ambient_stars",
        },
        "interaction_action": "Freely explore dialogue, metaverse portals, or idle chat.",
    },
}


class NPCAdaptiveController:
    """Tracks session emotional history and calculates adaptive interaction state."""

    def __init__(self, history_capacity: int = 15):
        self.history_capacity = history_capacity
        self.emotion_history: List[Dict[str, Any]] = []

    def record_interaction(self, emotion: str, confidence: float, modality: str = "multimodal") -> None:
        emotion = emotion.lower().strip()
        if emotion not in ADAPTATION_RULES:
            emotion = "neutral"
        self.emotion_history.append({
            "emotion": emotion,
            "confidence": confidence,
            "modality": modality,
        })
        if len(self.emotion_history) > self.history_capacity:
            self.emotion_history.pop(0)

    def get_current_adaptation(self, dominant_emotion: str, confidence: float) -> Dict[str, Any]:
        """Compute adaptation parameters given the current detected emotion."""
        emotion = dominant_emotion.lower().strip()
        rule = ADAPTATION_RULES.get(emotion, ADAPTATION_RULES["neutral"])

        # Detect emotional trend/persistence
        persistent_count = sum(1 for e in self.emotion_history[-4:] if e["emotion"] == emotion)
        high_intensity = confidence >= 0.70

        return {
            "dominant_emotion": emotion,
            "confidence": round(confidence, 4),
            "adaptation_style": rule["style"],
            "description": rule["description"],
            "tts_parameters": {
                "rate": rule["tts_rate"],
                "pitch": rule["tts_pitch"],
            },
            "environment_state": rule["environment"],
            "interaction_recommendation": rule["interaction_action"],
            "metaverse_context": {
                "high_intensity": high_intensity,
                "persistent_trend": persistent_count >= 3,
                "recent_history_length": len(self.emotion_history),
            },
        }


# Global singleton instance
adaptive_controller = NPCAdaptiveController()

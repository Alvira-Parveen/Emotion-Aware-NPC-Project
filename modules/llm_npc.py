"""
modules/llm_npc.py  —  Stage E: Generative AI / LLM NPC Response Layer
══════════════════════════════════════════════════════════════════════
Generates dynamic, context-aware dialogue for the NPC based on:
  1. The estimated emotional state (and confidence) from the fusion layer.
  2. The user's input text (or topic/utterance).
  3. Conversation history / context.
  4. Explicit NPC persona and behavioral constraints.

Key Principles & Safety:
  - Strict separation: Emotion detection is handled upstream; the LLM does NOT
    detect emotions, nor does its output validate emotion accuracy.
  - Ethical & respectful: The NPC never presumes to know the user's internal
    reality or psychological state with absolute certainty.
  - Fail-safe fallback: If API keys are missing, quota exceeded, or network fails,
    it automatically falls back to curated emotional dialogue templates.
  - Latency tracking: Every response measures round-trip inference latency (ms).
  - Secure: API keys are read strictly from environment variables.
"""

from __future__ import annotations

import os
import time
import logging
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

# Fallback dialogue bank matching web_app.py baseline
FALLBACK_DIALOGUES: Dict[str, List[str]] = {
    "happy": [
        "You radiate such warm energy! 😄 Keep that positive momentum going.",
        "Your bright mood is uplifting! ✨ What inspired your happiness today?",
        "It is wonderful to see you in high spirits! 🌟",
    ],
    "sad": [
        "I'm right here with you. 💙 Take things one step at a time.",
        "It's completely okay to feel down sometimes. 🤗 I am listening.",
        "Sending you calm thoughts. 🌈 Tomorrow brings fresh opportunities.",
    ],
    "angry": [
        "I sense frustration. Take a deep, slow breath with me... 🧘",
        "It's valid to feel upset, but let's navigate this calmly together. 🕊️",
        "Let's pause for a moment. Peace is your strongest anchor. ☮️",
    ],
    "surprise": [
        "That took us both by surprise! 😲 Tell me more about what happened.",
        "Unexpected moments keep things thrilling! 🎉",
        "Whoa! What a twist! 🤯",
    ],
    "fear": [
        "You are in a safe space here. 🛡️ Steady yourself, you're not alone.",
        "Take a slow breath. 🌿 We can face whatever is ahead together.",
        "I've got your back. 💪 Step by step, we will manage this.",
    ],
    "disgust": [
        "Something seems quite off-putting. 🤔 What rubbed you the wrong way?",
        "I see your distaste. 😅 Some things truly are unpleasant to deal with.",
        "Understandable reaction! 😬 Let's pivot to something more comfortable.",
    ],
    "neutral": [
        "Steady and grounded vibes. 😌 What would you like to explore today?",
        "Calm waters today. 🧊 I'm here whenever you wish to converse.",
        "At ease and centered. 🎧 How can I assist your journey?",
    ],
}

DEFAULT_FALLBACK = "I am here with you in this virtual space. How can I assist you today?"


def _format_system_prompt(persona_name: str = "Aria", adaptation_style: str = "supportive") -> str:
    return (
        f"You are {persona_name}, an empathetic, intelligent Non-Player Character (NPC) "
        "residing in an adaptive Metaverse virtual space. "
        "You are conversing directly with a user.\n\n"
        "Behavioral Guidelines:\n"
        "1. Speak naturally in 1 to 2 engaging, conversational sentences (suitable for speech output).\n"
        "2. Adjust your conversational tone according to the user's estimated emotional state and adaptation style.\n"
        "3. Empathy without presumption: Do NOT claim you read their mind or state absolute certainty about their personal feelings.\n"
        "4. Be respectful, supportive, immersive, and constructive.\n"
        "5. Avoid robotic preamble like 'As an AI' or 'Based on your emotion'. Jump right into immersive dialogue."
    )


def generate_response(
    user_message: str,
    detected_emotion: str = "neutral",
    emotion_confidence: float = 0.5,
    adaptation_style: str = "supportive",
    history: Optional[List[Dict[str, str]]] = None,
    persona_name: str = "Aria",
) -> Dict:
    """
    Generate an NPC response using an LLM (Gemini / OpenAI compatible) or fallback.

    Parameters
    ----------
    user_message : str
        The input text or utterance from the user.
    detected_emotion : str
        The fused or single-modality emotion label (e.g. 'happy', 'sad').
    emotion_confidence : float
        Confidence score between 0.0 and 1.0.
    adaptation_style : str
        Style directive from the adaptation layer (e.g., 'calming', 'uplifting', 'reflective').
    history : list of dicts, optional
        Past turns formatted as [{'role': 'user'|'npc', 'content': '...'}].
    persona_name : str
        NPC name.

    Returns
    -------
    dict with:
      reply: str
      source: 'gemini' | 'openai' | 'fallback_template'
      latency_ms: float
      emotion_used: str
      confidence_used: float
      adaptation_style: str
      error: str | None
    """
    start_time = time.time()
    detected_emotion = detected_emotion.lower().strip()
    if detected_emotion not in FALLBACK_DIALOGUES:
        detected_emotion = "neutral"

    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()

    # 1. Try Google Gemini API if key is present
    if gemini_key:
        try:
            import google.generativeai as genai  # type: ignore

            genai.configure(api_key=gemini_key)
            model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=_format_system_prompt(persona_name, adaptation_style),
            )

            prompt = (
                f"[Context: User estimated emotion is '{detected_emotion}' "
                f"(confidence: {emotion_confidence:.0%}). NPC tone directive: '{adaptation_style}'.]\n"
                f"User utterance: {user_message if user_message else '(User entered the space without speaking)'}\n"
                f"NPC Response:"
            )

            response = model.generate_content(prompt)
            reply_text = response.text.strip().replace('"', '')
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            return {
                "reply": reply_text,
                "source": f"gemini ({model_name})",
                "latency_ms": elapsed_ms,
                "emotion_used": detected_emotion,
                "confidence_used": emotion_confidence,
                "adaptation_style": adaptation_style,
                "error": None,
            }
        except Exception as e:
            logger.warning("Gemini generation failed: %s; falling back.", e)

    # 2. Try OpenAI API if key is present
    if openai_key:
        try:
            import openai  # type: ignore

            client = openai.OpenAI(api_key=openai_key)
            messages = [
                {"role": "system", "content": _format_system_prompt(persona_name, adaptation_style)},
                {
                    "role": "user",
                    "content": (
                        f"[Context: User estimated emotion is '{detected_emotion}' "
                        f"(confidence: {emotion_confidence:.0%}). NPC tone: '{adaptation_style}'.]\n"
                        f"{user_message if user_message else '(User enters room)'}"
                    ),
                },
            ]
            resp = client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo"),
                messages=messages,
                max_tokens=90,
                temperature=0.7,
            )
            reply_text = resp.choices[0].message.content.strip().replace('"', '')
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            return {
                "reply": reply_text,
                "source": "openai",
                "latency_ms": elapsed_ms,
                "emotion_used": detected_emotion,
                "confidence_used": emotion_confidence,
                "adaptation_style": adaptation_style,
                "error": None,
            }
        except Exception as e:
            logger.warning("OpenAI generation failed: %s; falling back.", e)

    # 3. Fallback: Rule & template-based dialogue bank
    import random
    choices = FALLBACK_DIALOGUES.get(detected_emotion, [DEFAULT_FALLBACK])
    reply_text = random.choice(choices)

    # If the user typed a specific message, prepend a gentle context acknowledgment
    if user_message and len(user_message.strip()) > 0:
        cleaned_msg = user_message.strip()
        if len(cleaned_msg) < 35:
            reply_text = f"Regarding '{cleaned_msg}' — {reply_text}"

    elapsed_ms = round((time.time() - start_time) * 1000, 2)
    return {
        "reply": reply_text,
        "source": "template_bank",
        "latency_ms": elapsed_ms,
        "emotion_used": detected_emotion,
        "confidence_used": emotion_confidence,
        "adaptation_style": adaptation_style,
        "error": None,
    }

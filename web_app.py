"""
🎭 Emotion-Aware NPC — Flask Web App
Uses browser's native webcam (getUserMedia) + FER backend.
Run:  python web_app.py
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
try:
    from fer import FER
except ImportError:
    FER = None
from PIL import Image, ImageDraw, ImageFont
import random
import base64
import io
import time
import os
import sys
import threading

# Add current directory to path for modular imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules import text_emotion, voice_emotion, fusion, llm_npc, npc_adapter

app = Flask(__name__)

# ─── Emotion detector (loaded lazily in background) ─────────────────
detector = None
model_loading = False

def load_model_background():
    global detector, model_loading, FER
    if FER is None:
        try:
            from fer import FER
        except Exception:
            pass
    if FER is None:
        print("ℹ️ Note: FER package not installed. Face detection requires 'fer'.", flush=True)
        return
    if detector is None and not model_loading:
        model_loading = True
        print("🔄 Loading emotion detection model in background...", flush=True)
        try:
            detector = FER(mtcnn=True)
            print("✅ Model loaded successfully!", flush=True)
        except Exception as e:
            print(f"❌ Error loading model: {e}", flush=True)
            model_loading = False

def get_detector():
    global FER
    if FER is None:
        try:
            from fer import FER
        except Exception:
            pass
    return detector

# ─── NPC Dialogue Bank ──────────────────────────────────────────────
dialogues = {
    "happy":    ["You look happy! 😄", "Nice smile! Keep it up! ✨", "Your joy is contagious! 🌟"],
    "sad":      ["Cheer up, it will be okay. 💙", "Stay strong, friend. 🤗", "Better days are coming. 🌈"],
    "angry":    ["Take a deep breath... 🧘", "Relax, it's alright. 🕊️", "Let it go, peace is power. ☮️"],
    "surprise": ["That's unexpected! 😲", "Wow, didn't see that coming! 🎉", "Surprised? Me too! 🤯"],
    "fear":     ["Don't worry, you're safe here. 🛡️", "It's okay, stay calm. 🌿", "Breathe... I've got your back. 💪"],
    "disgust":  ["Something bothering you? 🤔", "Not a fan, huh? 😅", "I get it, some things are just... ew. 😬"],
    "neutral":  ["All good. 😌", "Just chilling, I see. 🧊", "Steady vibes. 🎧"],
}

npc_avatars = {
    "happy": "😊", "sad": "😢", "angry": "😠",
    "surprise": "😲", "fear": "😨", "disgust": "🤢", "neutral": "😐",
}

emotion_colors = {
    "happy":    "#48c78e", "sad":      "#6495ed", "angry":    "#e74c3c",
    "surprise": "#f1c40f", "fear":     "#9b59b6", "disgust":  "#2ecc71",
    "neutral":  "#95a5a6",
}

emotion_colors_rgb = {
    "happy":    (72, 199, 142), "sad":      (100, 149, 237), "angry":    (231, 76, 60),
    "surprise": (241, 196, 15), "fear":     (155, 89, 182),  "disgust":  (46, 204, 113),
    "neutral":  (149, 165, 166),
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    data = request.json
    image_data = data.get("image", "")

    # Decode base64 image
    try:
        header, b64 = image_data.split(",", 1)
        img_bytes = base64.b64decode(b64)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        frame = np.array(pil_img)
    except Exception as e:
        return jsonify({"error": f"Could not decode image: {str(e)}"}), 400

    # Detect emotions (ALL faces)
    det = get_detector()
    if det is None:
        if FER is None:
            return jsonify({"error": "Facial emotion detection requires 'fer' (pip install fer). You can still test Text, Voice, and Multimodal!"})
        # Start background load if not already started
        threading.Thread(target=load_model_background, daemon=True).start()
        return jsonify({"error": "⏳ Model is warming up! This takes ~2 minutes. Please wait and try again."})

    results = det.detect_emotions(frame)

    if not results:
        return jsonify({"error": "No face detected — try better lighting or look straight at the camera!"})

    # Draw on ALL detected faces
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        except (OSError, IOError):
            font = ImageFont.load_default()

    faces_data = []
    for i, face in enumerate(results):
        emotions = face["emotions"]
        dominant = max(emotions, key=emotions.get)
        confidence = emotions[dominant]

        (x, y, w, h) = face["box"]
        color = emotion_colors_rgb.get(dominant, (255, 255, 255))

        # Glow effect
        for offset in range(3, 0, -1):
            glow = tuple(min(255, c + 60) for c in color)
            draw.rectangle([x - offset, y - offset, x + w + offset, y + h + offset], outline=glow, width=1)
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)

        # Label
        label = f"#{i+1} {dominant.upper()} ({confidence:.0%})"
        bbox_text = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox_text[2] - bbox_text[0], bbox_text[3] - bbox_text[1]
        draw.rectangle([x, y - th - 10, x + tw + 10, y], fill=color)
        draw.text((x + 5, y - th - 8), label, fill=(255, 255, 255), font=font)

        avatar = npc_avatars.get(dominant, "🤖")
        reply = random.choice(dialogues.get(dominant, ["..."]))

        faces_data.append({
            "id": i + 1,
            "emotion": dominant.upper(),
            "confidence": f"{confidence:.0%}",
            "confidence_raw": confidence,
            "emotions": emotions,
            "avatar": avatar,
            "npc_reply": reply,
            "color": emotion_colors.get(dominant, "#888"),
        })

    # Encode result image
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    result_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Primary face (highest confidence) for backward compatibility
    primary = max(faces_data, key=lambda f: f["confidence_raw"])

    return jsonify({
        "emotion": primary["emotion"],
        "confidence": primary["confidence"],
        "emotions": primary["emotions"],
        "avatar": primary["avatar"],
        "npc_reply": primary["npc_reply"],
        "color": primary["color"],
        "result_image": result_b64,
        "face_count": len(faces_data),
        "faces": faces_data,
    })


# ─── Challenge Mode Endpoint ────────────────────────────────────────
CHALLENGE_EMOTIONS = ["happy", "sad", "angry", "surprise", "fear", "neutral"]

@app.route("/challenge", methods=["POST"])
def challenge():
    """Score how well the user matches a target emotion."""
    data = request.json
    image_data = data.get("image", "")
    target = data.get("target", "happy").lower()

    try:
        header, b64 = image_data.split(",", 1)
        img_bytes = base64.b64decode(b64)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        frame = np.array(pil_img)
    except Exception as e:
        return jsonify({"error": f"Could not decode image: {str(e)}"}), 400

    det = get_detector()
    if det is None:
        threading.Thread(target=load_model_background, daemon=True).start()
        return jsonify({"error": "⏳ Model is warming up! This takes ~2 minutes. Please wait and try again."})

    results = det.detect_emotions(frame)
    if not results:
        return jsonify({"error": "No face detected! Make sure your face is visible."})

    face = results[0]
    emotions = face["emotions"]
    score = emotions.get(target, 0)
    dominant = max(emotions, key=emotions.get)

    # Feedback based on score
    if score >= 0.7:
        feedback = "🏆 PERFECT! You nailed it!"
        grade = "S"
    elif score >= 0.5:
        feedback = "🌟 Great job! Very convincing!"
        grade = "A"
    elif score >= 0.3:
        feedback = "👍 Good effort! Getting there!"
        grade = "B"
    elif score >= 0.15:
        feedback = "🤔 Not quite... try harder!"
        grade = "C"
    else:
        feedback = "😅 That doesn't look like " + target + " at all!"
        grade = "D"

    return jsonify({
        "target": target,
        "score": round(score * 100, 1),
        "grade": grade,
        "feedback": feedback,
        "actual_emotion": dominant,
        "emotions": emotions,
    })


@app.route("/challenge/random")
def random_challenge():
    """Get a random emotion to challenge the user."""
    emotion = random.choice(CHALLENGE_EMOTIONS)
    prompts = {
        "happy":    "Show me your happiest smile! 😄",
        "sad":      "Look as sad as you can! 😢",
        "angry":    "Give me your fiercest angry face! 😠",
        "surprise": "Act totally surprised! 😲",
        "fear":     "Show me a scared expression! 😨",
        "neutral":  "Keep a completely straight face! 😐",
    }
    return jsonify({
        "emotion": emotion,
        "prompt": prompts.get(emotion, f"Show me {emotion}!"),
    })


# ─── Modular Multimodal Pipeline Endpoints ─────────────────────────

def _run_face_detection_internal(image_data: str):
    """Internal helper to process a face frame and return structured results."""
    t0 = time.perf_counter()
    if not image_data:
        return None, "No image data provided"

    try:
        if "," in image_data:
            header, b64 = image_data.split(",", 1)
        else:
            b64 = image_data
        img_bytes = base64.b64decode(b64)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        frame = np.array(pil_img)
    except Exception as e:
        return None, f"Image decode failed: {str(e)}"

    det = get_detector()
    if det is None:
        if FER is None:
            return None, "Face detection requires 'fer' (pip install fer). Testing text, voice, and multimodal fusion remains fully operational."
        threading.Thread(target=load_model_background, daemon=True).start()
        return None, "Face emotion model is warming up (~2 min). Please retry shortly."

    results = det.detect_emotions(frame)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    if not results:
        return None, "No face detected in the provided image frame."

    primary_face = results[0]
    emotions = primary_face["emotions"]
    dominant = max(emotions, key=emotions.get)
    confidence = float(emotions[dominant])

    return {
        "emotion": dominant,
        "confidence": confidence,
        "scores": emotions,
        "backend": "fer_mtcnn",
        "box": primary_face.get("box"),
        "latency_ms": latency_ms,
        "face_count": len(results),
    }, None


@app.route("/detect_text", methods=["POST"])
def detect_text():
    """Stage B: Text emotion recognition endpoint with adaptive response."""
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()

    t0 = time.perf_counter()
    pred = text_emotion.predict(text)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    pred["latency_ms"] = latency_ms

    dominant = pred.get("emotion", "neutral")
    conf = pred.get("confidence", 0.0)

    # Stage F: Adaptation
    adaptation = npc_adapter.adaptive_controller.get_current_adaptation(dominant, conf)
    npc_adapter.adaptive_controller.record_interaction(dominant, conf, modality="text")

    # Stage E: Response generation
    llm_resp = llm_npc.generate_response(
        user_message=text,
        detected_emotion=dominant,
        emotion_confidence=conf,
        adaptation_style=adaptation["adaptation_style"],
    )

    return jsonify({
        "modality": "text",
        "prediction": pred,
        "adaptation": adaptation,
        "npc_response": llm_resp,
        "avatar": npc_avatars.get(dominant, "🤖"),
        "color": emotion_colors.get(dominant, "#95a5a6"),
    })


@app.route("/detect_voice", methods=["POST"])
def detect_voice():
    """Stage C: Voice emotion recognition from raw acoustic characteristics."""
    audio_bytes = None
    mime_hint = "audio/webm"

    if "audio" in request.files:
        audio_file = request.files["audio"]
        audio_bytes = audio_file.read()
        mime_hint = audio_file.content_type or "audio/webm"
    elif request.is_json:
        data = request.get_json()
        b64_audio = data.get("audio", "")
        mime_hint = data.get("mime", "audio/webm")
        if "," in b64_audio:
            _, b64_audio = b64_audio.split(",", 1)
        if b64_audio:
            try:
                audio_bytes = base64.b64decode(b64_audio)
            except Exception as e:
                return jsonify({"error": f"Invalid audio base64: {e}"}), 400

    if not audio_bytes:
        return jsonify({"error": "No audio received. Record audio or upload an audio file."}), 400

    t0 = time.perf_counter()
    pred = voice_emotion.predict(audio_bytes, mime_hint=mime_hint)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    pred["latency_ms"] = latency_ms

    dominant = pred.get("emotion", "neutral")
    conf = pred.get("confidence", 0.0)

    # Stage F: Adaptation
    adaptation = npc_adapter.adaptive_controller.get_current_adaptation(dominant, conf)
    npc_adapter.adaptive_controller.record_interaction(dominant, conf, modality="voice")

    # Stage E: Response generation
    llm_resp = llm_npc.generate_response(
        user_message="",
        detected_emotion=dominant,
        emotion_confidence=conf,
        adaptation_style=adaptation["adaptation_style"],
    )

    return jsonify({
        "modality": "voice",
        "prediction": pred,
        "adaptation": adaptation,
        "npc_response": llm_resp,
        "avatar": npc_avatars.get(dominant, "🤖"),
        "color": emotion_colors.get(dominant, "#95a5a6"),
    })


@app.route("/detect_multimodal", methods=["POST"])
def detect_multimodal():
    """
    Stage D: Multimodal Emotion Fusion (Face + Voice + Text).
    Takes any combination of face image, voice audio, and text input,
    executes modality predictions, runs score-level fusion,
    applies adaptive environment tokens, and generates an LLM/fallback response.
    """
    total_t0 = time.perf_counter()
    data = request.get_json(silent=True) or {}

    image_data = data.get("image", "")
    text_data = data.get("text", "").strip()
    audio_data = data.get("audio", "")
    audio_mime = data.get("audio_mime", "audio/webm")
    custom_weights = data.get("weights")

    latencies = {}

    # 1. Modality: Face
    face_result = None
    if image_data:
        face_res, face_err = _run_face_detection_internal(image_data)
        if face_res:
            face_result = face_res
            latencies["face_ms"] = face_res.get("latency_ms", 0.0)
        else:
            face_result = {"error": face_err, "emotion": "neutral", "confidence": 0.0, "scores": {}}

    # 2. Modality: Text
    text_result = None
    if text_data:
        t0 = time.perf_counter()
        text_result = text_emotion.predict(text_data)
        latencies["text_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        text_result["latency_ms"] = latencies["text_ms"]

    # 3. Modality: Voice
    voice_result = None
    if audio_data:
        t0 = time.perf_counter()
        try:
            if "," in audio_data:
                _, b64_a = audio_data.split(",", 1)
            else:
                b64_a = audio_data
            raw_audio = base64.b64decode(b64_a)
            voice_result = voice_emotion.predict(raw_audio, mime_hint=audio_mime)
        except Exception as e:
            voice_result = {"error": f"Audio processing error: {e}", "emotion": "neutral", "confidence": 0.0, "scores": {}}
        latencies["voice_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        voice_result["latency_ms"] = latencies["voice_ms"]

    # 4. Stage D: Multimodal Fusion
    t0 = time.perf_counter()
    fused = fusion.fuse(
        face_result=face_result,
        text_result=text_result,
        voice_result=voice_result,
        weights=custom_weights,
    )
    latencies["fusion_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    dominant = fused["emotion"]
    confidence = fused["confidence"]

    # 5. Stage F: Adaptation
    t0 = time.perf_counter()
    adaptation = npc_adapter.adaptive_controller.get_current_adaptation(dominant, confidence)
    npc_adapter.adaptive_controller.record_interaction(dominant, confidence, modality="multimodal")
    latencies["adaptation_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # 6. Stage E: LLM / Dialogue Generation
    t0 = time.perf_counter()
    llm_resp = llm_npc.generate_response(
        user_message=text_data,
        detected_emotion=dominant,
        emotion_confidence=confidence,
        adaptation_style=adaptation["adaptation_style"],
    )
    latencies["llm_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    latencies["end_to_end_ms"] = round((time.perf_counter() - total_t0) * 1000, 2)

    return jsonify({
        "status": "success",
        "fused_emotion": dominant,
        "fused_confidence": confidence,
        "fused_scores": fused["scores"],
        "fusion_details": fused,
        "modalities": {
            "face": face_result,
            "text": text_result,
            "voice": voice_result,
        },
        "adaptation": adaptation,
        "npc_response": llm_resp,
        "avatar": npc_avatars.get(dominant, "🤖"),
        "color": emotion_colors.get(dominant, "#95a5a6"),
        "latencies": latencies,
    })


@app.route("/config", methods=["GET"])
def get_config():
    """Return runtime configuration, fusion weights, and active model backends."""
    return jsonify({
        "fusion_weights": fusion.get_default_weights(),
        "text_backend": text_emotion.ensure_loaded(),
        "voice_backend": voice_emotion.ensure_loaded(),
        "llm_gemini_configured": bool(os.environ.get("GEMINI_API_KEY")),
        "llm_openai_configured": bool(os.environ.get("OPENAI_API_KEY")),
        "supported_emotions": fusion.ALL_EMOTIONS,
    })


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  🎭 Emotion NPC is running!")
    print("  🖥️  Local:  http://127.0.0.1:8080")
    print("=" * 55 + "\n")
    # Pre-warm FER detector in background
    threading.Thread(target=load_model_background, daemon=True).start()
    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "0.0.0.0")
    app.run(host=host, port=port, debug=False)



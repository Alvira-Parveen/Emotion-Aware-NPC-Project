"""
🎭 Emotion-Aware NPC — Flask Web App
Uses browser's native webcam (getUserMedia) + FER backend.
Run:  python web_app.py
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
from fer import FER
from PIL import Image, ImageDraw, ImageFont
import random
import base64
import io

app = Flask(__name__)

# ─── Emotion detector (loaded lazily) ─────────────────────────────────
detector = None

def get_detector():
    global detector
    if detector is None:
        print("🔄 Loading emotion detection model...", flush=True)
        # This takes a long time on Render's free tier
        detector = FER(mtcnn=True)
        print("✅ Model loaded!", flush=True)
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


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  🎭 Emotion NPC is running!")
    print("  🖥️  Local:  http://127.0.0.1:8080")
    print("=" * 55 + "\n")
    app.run(host="0.0.0.0", port=8080, debug=False)


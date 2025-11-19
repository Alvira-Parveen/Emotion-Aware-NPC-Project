import cv2
from fer import FER
import time
import random
import subprocess
import threading

# Emotion detector
detector = FER(mtcnn=True)

# Last spoken time
last_spoken = 0
cooldown = 5.0  # seconds between NPC voice

# Simple, clean dialogues
dialogues = {
    "happy": ["You look happy!", "Nice smile!"],
    "sad": ["Cheer up, it will be okay.", "Stay strong."],
    "angry": ["Take a deep breath.", "Relax, it’s alright."],
    "surprise": ["That’s unexpected!", "Wow!"],
    "fear": ["Don’t worry, you’re safe.", "It’s okay, stay calm."],
    "neutral": ["All good.", "Just chilling."]
}

# --- Non-blocking voice function ---
def npc_speak(text):
    threading.Thread(
        target=lambda: subprocess.run(["say", text])
    ).start()

# --- Main loop ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not access the webcam.")
    exit()

print("🎥 Webcam started — press 'q' in the window to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect emotions
    results = detector.detect_emotions(frame)

    if results:
        # Use the first face detected
        face = results[0]
        emotions = face["emotions"]
        emotion = max(emotions, key=emotions.get)

        # Draw rectangle + label
        (x, y, w, h) = face["box"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Speak only every cooldown seconds
        now = time.time()
        if now - last_spoken > cooldown:
            last_spoken = now
            reply = random.choice(dialogues.get(emotion, ["..."]))
            print(f"🧙 NPC ({emotion}): {reply}")
            npc_speak(reply)

    cv2.imshow("NPC Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("NPC session ended.")

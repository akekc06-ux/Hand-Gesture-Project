import cv2
import time
import numpy as np
import mediapipe as mp

# ─────────────────────────────────────────────
#  MediaPipe setup  (max 2 hands enforced here)
# ─────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)
# ─────────────────────────────────────────────
#  Landmark indices
# ─────────────────────────────────────────────
TIPS = [
    mp_hands.HandLandmark.THUMB_TIP,
    mp_hands.HandLandmark.INDEX_FINGER_TIP,
    mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
    mp_hands.HandLandmark.RING_FINGER_TIP,
    mp_hands.HandLandmark.PINKY_TIP,
]
PIPS = [
    mp_hands.HandLandmark.THUMB_IP,
    mp_hands.HandLandmark.INDEX_FINGER_PIP,
    mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
    mp_hands.HandLandmark.RING_FINGER_PIP,
    mp_hands.HandLandmark.PINKY_PIP,
]

# ─────────────────────────────────────────────
#  Gesture classifier
# ─────────────────────────────────────────────
def fingers_up(hand_landmarks, handedness_label):
    """Return list of 5 booleans - True if finger is extended."""
    lm = hand_landmarks.landmark
    up = []
    # Thumb: x-axis comparison (mirror-aware)
    if handedness_label == "Right":
        up.append(lm[TIPS[0]].x < lm[PIPS[0]].x)
    else:
        up.append(lm[TIPS[0]].x > lm[PIPS[0]].x)
    # Other 4 fingers: tip above pip = extended
    for i in range(1, 5):
        up.append(lm[TIPS[i]].y < lm[PIPS[i]].y)
    return up


def classify_gesture(hand_landmarks, handedness_label):
    """Return (gesture_name, emoji)."""
    up = fingers_up(hand_landmarks, handedness_label)
    if not any(up[1:]):
        return "Rock",     "Rock"
    if all(up[1:]):
        return "Paper",    "Paper"
    if up[1] and up[2] and not up[3] and not up[4]:
        return "Scissors", "Scissors"
    return "Unknown",  "Unknown"


# ─────────────────────────────────────────────
#  Game logic
# ─────────────────────────────────────────────
BEATS = {
    ("Rock",     "Scissors"): 0,
    ("Scissors", "Paper"):    0,
    ("Paper",    "Rock"):     0,
    ("Scissors", "Rock"):     1,
    ("Paper",    "Scissors"): 1,
    ("Rock",     "Paper"):    1,
}

def determine_winner(g1, g2):
    """0=P1 wins, 1=P2 wins, -1=Tie, None=unknown."""
    if "Unknown" in (g1, g2):
        return None
    if g1 == g2:
        return -1
    return BEATS.get((g1, g2))


# ─────────────────────────────────────────────
#  Drawing helpers
# ─────────────────────────────────────────────
C = {
    "P1":     (255, 180,  50),
    "P2":     ( 80, 200, 255),
    "tie":    (200, 200, 200),
    "bg":     ( 18,  18,  28),
    "accent": (180,  80, 255),
}
FONT  = cv2.FONT_HERSHEY_DUPLEX
FONTB = cv2.FONT_HERSHEY_TRIPLEX


def overlay_rect(img, x, y, w, h, color, alpha=0.55):
    ov = img.copy()
    cv2.rectangle(ov, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


def centered(img, text, cy, font, scale, color, thick=2):
    (tw, _), _ = cv2.getTextSize(text, font, scale, thick)
    tx = (img.shape[1] - tw) // 2
    cv2.putText(img, text, (tx, cy), font, scale, color, thick, cv2.LINE_AA)


# ─────────────────────────────────────────────
#  Main loop
# ─────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    scores = [0, 0]
    ties   = 0

    last_result   = ""
    last_gestures = ["", ""]
    result_color  = C["tie"]

    JUDGE_EVERY   = 2.0
    STABLE_NEEDED = 10
    last_judge_t  = time.time()
    stable_counts = [0, 0]
    prev_gestures = ["", ""]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)
        detected = {}

        if result.multi_hand_landmarks:
            for lm, hd in zip(result.multi_hand_landmarks,
                              result.multi_handedness):
                label  = hd.classification[0].label
                g_name, g_disp = classify_gesture(lm, label)
                player = "P1" if label == "Left" else "P2"
                detected[player] = (g_name, g_disp, lm)

                mp_drawing.draw_landmarks(
                    frame, lm, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

        # Stability tracking
        cur = [detected.get("P1", ("", "", None))[0],
               detected.get("P2", ("", "", None))[0]]
        for i in range(2):
            if cur[i] and cur[i] == prev_gestures[i]:
                stable_counts[i] += 1
            else:
                stable_counts[i]  = 0
        prev_gestures = cur[:]

        both_stable = (stable_counts[0] >= STABLE_NEEDED and
                       stable_counts[1] >= STABLE_NEEDED)

        # Auto-judge
        now = time.time()
        if both_stable and (now - last_judge_t) >= JUDGE_EVERY:
            g1     = cur[0]
            g2     = cur[1]
            winner = determine_winner(g1, g2)
            last_gestures = [
                detected.get("P1", ("?", "?", None))[0],
                detected.get("P2", ("?", "?", None))[0],
            ]
            if winner == 0:
                scores[0]   += 1
                last_result  = "Player 1 Wins!"
                result_color = C["P1"]
            elif winner == 1:
                scores[1]   += 1
                last_result  = "Player 2 Wins!"
                result_color = C["P2"]
            elif winner == -1:
                ties        += 1
                last_result  = "It's a Tie!"
                result_color = C["tie"]
            last_judge_t  = now
            stable_counts = [0, 0]

        # ─── UI ───────────────────────────────────
        # Top bar
        overlay_rect(frame, 0, 0, w, 90, C["bg"], alpha=0.75)
        cv2.putText(frame, "ROCK  PAPER  SCISSORS",
                    (w // 2 - 225, 38), FONTB, 0.9, C["accent"], 2, cv2.LINE_AA)
        cv2.putText(frame, f"P1 (Left):  {scores[0]}",
                    (20, 75), FONT, 0.7, C["P1"], 2, cv2.LINE_AA)
        cv2.putText(frame, f"P2 (Right): {scores[1]}",
                    (w - 265, 75), FONT, 0.7, C["P2"], 2, cv2.LINE_AA)
        cv2.putText(frame, f"Ties: {ties}",
                    (w // 2 - 45, 75), FONT, 0.65, C["tie"], 1, cv2.LINE_AA)

        # Per-player badge
        for idx, (player, color, bx) in enumerate([
            ("P1", C["P1"], 20),
            ("P2", C["P2"], w - 220),
        ]):
            data = detected.get(player)
            if data:
                g_name, _, _ = data
                overlay_rect(frame, bx, h - 140, 200, 120, color, alpha=0.22)
                cv2.putText(frame, player,
                            (bx + 10, h - 105), FONTB, 0.8, color, 2, cv2.LINE_AA)
                cv2.putText(frame, g_name,
                            (bx + 10, h - 68), FONT, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
                bar_fill = int(min(stable_counts[idx], STABLE_NEEDED) / STABLE_NEEDED * 180)
                cv2.rectangle(frame, (bx + 10, h - 50), (bx + 190, h - 34), (60,60,60), -1)
                cv2.rectangle(frame, (bx + 10, h - 50), (bx + 10 + bar_fill, h - 34), color, -1)
                label_txt = "LOCKED" if stable_counts[idx] >= STABLE_NEEDED else "LOCKING..."
                cv2.putText(frame, label_txt,
                            (bx + 10, h - 18), FONT, 0.45, color, 1, cv2.LINE_AA)
            else:
                overlay_rect(frame, bx, h - 140, 200, 120, (60,60,60), alpha=0.22)
                cv2.putText(frame, player,
                            (bx + 10, h - 105), FONTB, 0.8, (120,120,120), 2, cv2.LINE_AA)
                cv2.putText(frame, "No hand",
                            (bx + 10, h - 68), FONT, 0.75, (100,100,100), 1, cv2.LINE_AA)

        # Result banner
        if last_result:
            banner = f"{last_result}  |  {last_gestures[0]} vs {last_gestures[1]}"
            overlay_rect(frame, w // 2 - 320, h // 2 - 42, 640, 68, (20,20,20), alpha=0.82)
            centered(frame, banner, h // 2 + 8, FONTB, 0.75, result_color, 2)

        # Bottom bar
        overlay_rect(frame, 0, h - 30, w, 30, C["bg"], alpha=0.65)
        cv2.putText(frame,
                    "  Q: Quit   R: Reset   SPACE: Force judge   "
                    "Hold gesture steady to lock -> auto-judges every 2s",
                    (10, h - 9), FONT, 0.42, (160,160,160), 1, cv2.LINE_AA)

        cv2.imshow("Rock Paper Scissors", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            scores, ties = [0, 0], 0
            last_result  = ""
            print("Scores reset.")
        elif key == ord(' ') and both_stable:
            # Force immediate judge
            last_judge_t = 0

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinal -- P1: {scores[0]}  P2: {scores[1]}  Ties: {ties}")


if __name__ == "__main__":
    main()
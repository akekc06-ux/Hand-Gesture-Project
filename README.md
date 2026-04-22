# ✋ Rock Paper Scissors (Hand Gesture Game)

A real-time two-player **Rock-Paper-Scissors game** powered by **computer vision**, using **OpenCV** and **MediaPipe** to detect and classify hand gestures through a webcam.

---

## 🚀 Features

* 🎥 Real-time hand tracking using webcam
* 🤖 Gesture recognition (Rock, Paper, Scissors)
* 👥 Two-player mode (Left hand = Player 1, Right hand = Player 2)
* 🎯 Auto gesture detection with stability check
* 🧠 Smart winner detection logic
* 📊 Live scoreboard with ties tracking
* 🎨 Clean UI overlay with animations and feedback

---

## 🛠️ Tech Stack

* Python
* OpenCV
* MediaPipe
* NumPy

---

## 📂 Project Structure

```
├── hand_gesture_project.py
└── README.md
```

---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/rock-paper-scissors-cv.git
cd rock-paper-scissors-cv
```

2. Install dependencies:

```bash
pip install opencv-python mediapipe numpy
```

---

## ▶️ How to Run

```bash
python hand_gesture_project.py
```

---

## 🎮 Controls

* **Q** → Quit the game
* **R** → Reset scores
* **SPACE** → Force result (if both gestures are stable)

---

## 🧠 How It Works

* MediaPipe detects hand landmarks
* Finger positions determine gesture:

  * ✊ Rock → No fingers up
  * ✋ Paper → All fingers up
  * ✌️ Scissors → Index + middle fingers up
* Stability tracking ensures gestures are consistent before judging
* Game auto-evaluates every 2 seconds

---

## 💡 Future Improvements

* Add sound effects 🔊
* Single-player vs AI 🤖
* Gesture customization
* GUI-based menu system

---

## 🤝 Contributing

Pull requests are welcome. For major changes, open an issue first to discuss what you'd like to improve.

---

## 📜 License

This project is open-source.

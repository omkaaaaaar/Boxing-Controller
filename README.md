# 🥊 Virtual Boxing Controller – Play & Sweat!

Control [**Punchers**](https://poki.com/en/g/punchers) — a free browser boxing game — **using your real punches, hooks, and ducks!**

This project turns your body into a game controller using your webcam, powered by **OpenCV**, **MediaPipe**, and **pyautogui**. A fun way to sweat it out while gaming!

----

## 🎮 What It Does

| Real-World Move       | In-Game Key Press | Action Triggered     |
|-----------------------|-------------------|-----------------------|
| Straight left punch   | `R`               | **Jab**               |
| Straight right punch  | `T` (0.05s hold)  | **Hook**              |
| Duck your head (hand) | `A` (hold)        | **Duck**              |

Just stand in front of your laptop, launch the script, and start punching — your movements control the game directly!

---

## 🔧 How It Works

- **MediaPipe Pose** detects your body joints in real time.
- **Arm angles** are calculated to determine jab or hook.
- **Relative hand-to-head position** is used to detect a duck.
- **PyAutoGUI** simulates keypresses (`R`, `T`, `A`) based on the moves.

---

## 🏃 Why Use It?

> 🎮 + 💪 = 🥵

- Skip the keyboard — **fight with your fists**
- Have fun while doing a quick cardio workout
- Turn any browser boxing game into an interactive fitness game
- Works great with [Punchers on Poki](https://poki.com/en/g/punchers)

---

## 📦 Requirements

- Python 3.7+
- Webcam
- Works on macOS, Windows, Linux

### 🛠 Install Dependencies

```bash
pip install opencv-python mediapipe pyautogui

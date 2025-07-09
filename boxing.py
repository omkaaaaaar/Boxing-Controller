import cv2
import mediapipe as mp
import pyautogui
import math
import time

# MediaPipe pose setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 30)

# Timing & flags
jab_held = False
hook_held = False
hook_time = 0
duck_held = False
last_log_time = 0

# Colors
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)

def get_coords(landmark, shape):
    h, w = shape[:2]
    return int(landmark.x * w), int(landmark.y * h)

def calculate_angle(a, b, c):
    ax, ay = a.x, a.y
    bx, by = b.x, b.y
    cx, cy = c.x, c.y
    ab = [ax - bx, ay - by]
    cb = [cx - bx, cy - by]
    dot = ab[0]*cb[0] + ab[1]*cb[1]
    mag_ab = math.hypot(*ab)
    mag_cb = math.hypot(*cb)
    if mag_ab * mag_cb == 0:
        return 0
    return math.degrees(math.acos(min(1.0, max(-1.0, dot / (mag_ab * mag_cb)))))

def log_event(msg):
    global last_log_time
    if time.time() - last_log_time >= 0.2:
        print(msg)
        last_log_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    now = time.time()

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # ✅ FRONT HAND (closer to camera) = LEFT landmarks = HOOK = RED
        front_wrist, front_elbow, front_shoulder = lm[mp_pose.PoseLandmark.LEFT_WRIST], lm[mp_pose.PoseLandmark.LEFT_ELBOW], lm[mp_pose.PoseLandmark.LEFT_SHOULDER]

        # ✅ BACK HAND (farther) = RIGHT landmarks = JAB = BLUE
        back_wrist, back_elbow, back_shoulder = lm[mp_pose.PoseLandmark.RIGHT_WRIST], lm[mp_pose.PoseLandmark.RIGHT_ELBOW], lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        nose = lm[mp_pose.PoseLandmark.NOSE]

        # Angles
        hook_angle = calculate_angle(front_shoulder, front_elbow, front_wrist)
        jab_angle = calculate_angle(back_shoulder, back_elbow, back_wrist)

        # === JAB (Back arm) → R
        if jab_angle > 120 and not jab_held:
            pyautogui.press('r')
            jab_held = True
            log_event("JAB 👊")
        elif jab_angle <= 120:
            jab_held = False

        # === HOOK (Front arm) → T + W
        if hook_angle > 150 and front_elbow.y < front_shoulder.y + 0.1 and not hook_held:
            pyautogui.keyDown('t')
            pyautogui.keyDown('w')
            hook_held = True
            hook_time = now
            log_event("HOOK 🥊")

        if hook_held and now - hook_time >= 0.05:
            pyautogui.keyUp('t')
            pyautogui.keyUp('w')
            hook_held = False

        # === DUCK → A (one or both hands above nose)
        if back_wrist.y < nose.y or front_wrist.y < nose.y:
            if not duck_held:
                pyautogui.keyDown('a')
                duck_held = True
                log_event("DUCK 🦆")
        else:
            if duck_held:
                pyautogui.keyUp('a')
                duck_held = False

        # Draw back (blue) = JAB
        for point in [back_shoulder, back_elbow, back_wrist]:
            cv2.circle(frame, get_coords(point, frame.shape), 5, COLOR_BLUE, -1)
        cv2.line(frame, get_coords(back_shoulder, frame.shape), get_coords(back_elbow, frame.shape), COLOR_BLUE, 2)
        cv2.line(frame, get_coords(back_elbow, frame.shape), get_coords(back_wrist, frame.shape), COLOR_BLUE, 2)

        # Draw front (red) = HOOK
        for point in [front_shoulder, front_elbow, front_wrist]:
            cv2.circle(frame, get_coords(point, frame.shape), 5, COLOR_RED, -1)
        cv2.line(frame, get_coords(front_shoulder, frame.shape), get_coords(front_elbow, frame.shape), COLOR_RED, 2)
        cv2.line(frame, get_coords(front_elbow, frame.shape), get_coords(front_wrist, frame.shape), COLOR_RED, 2)

    cv2.imshow("Virtual Boxing", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

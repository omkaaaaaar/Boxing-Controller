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
        # Left = back = hook = RED
        rw, re, rs = lm[mp_pose.PoseLandmark.LEFT_WRIST], lm[mp_pose.PoseLandmark.LEFT_ELBOW], lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        # Right = front = jab = BLUE
        lw, le, ls = lm[mp_pose.PoseLandmark.RIGHT_WRIST], lm[mp_pose.PoseLandmark.RIGHT_ELBOW], lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        nose = lm[mp_pose.PoseLandmark.NOSE]

        # Angles
        right_arm_angle = calculate_angle(ls, le, lw)  # Jab (Right)
        left_arm_angle = calculate_angle(rs, re, rw)   # Hook (Left)

        # === JAB (Right arm) → R
        if right_arm_angle > 120 and not jab_held:
            pyautogui.press('r')
            jab_held = True
            log_event("JAB 👊")
        elif right_arm_angle <= 120:
            jab_held = False

        # === FIXED HOOK: Arm extended (angle-based) + elbow pushed forward or straight
        if left_arm_angle > 150 and re.y < rs.y + 0.1 and not hook_held:
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
        if lw.y < nose.y or rw.y < nose.y:
            if not duck_held:
                pyautogui.keyDown('a')
                duck_held = True
                log_event("DUCK 🦆")
        else:
            if duck_held:
                pyautogui.keyUp('a')
                duck_held = False

        # Draw arms
        for point in [ls, le, lw]:
            cv2.circle(frame, get_coords(point, frame.shape), 5, COLOR_BLUE, -1)
        for point in [rs, re, rw]:
            cv2.circle(frame, get_coords(point, frame.shape), 5, COLOR_RED, -1)
        cv2.line(frame, get_coords(ls, frame.shape), get_coords(le, frame.shape), COLOR_BLUE, 2)
        cv2.line(frame, get_coords(le, frame.shape), get_coords(lw, frame.shape), COLOR_BLUE, 2)
        cv2.line(frame, get_coords(rs, frame.shape), get_coords(re, frame.shape), COLOR_RED, 2)
        cv2.line(frame, get_coords(re, frame.shape), get_coords(rw, frame.shape), COLOR_RED, 2)

    cv2.imshow("Virtual Boxing", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

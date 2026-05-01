import cv2
import numpy as np
import mediapipe as mp

# ---------- MEDIAPIPE SETUP ----------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

counter = 0
stage = "UP"
angle = 0


# ---------- ANGLE FUNCTION ----------
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])

    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle


# ---------- START CAMERA ----------
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Convert BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Pose detection
    results = pose.process(rgb)

    if results.pose_landmarks:

        landmarks = results.pose_landmarks.landmark

        hip = [
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
        ]

        knee = [
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y
        ]

        ankle = [
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
        ]

        # Calculate angle
        angle = calculate_angle(hip, knee, ankle)

        # ---------- REP LOGIC ----------
        if angle > 150:
            if stage == "DOWN":
                counter += 1
                print("Rep:", counter)
            stage = "UP"

        if angle < 120:
            stage = "DOWN"

        # ---------- DRAW SKELETON ----------
        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        # Show angle near knee
        cv2.putText(
            frame,
            str(int(angle)),
            tuple(np.multiply(knee, [640, 480]).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    # ---------- MODERN UI ----------
    cv2.rectangle(frame, (0, 0), (300, 120), (40, 40, 40), -1)

    # Title
    cv2.putText(frame, "ATHLETE AI", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 255), 2)

    # Reps
    cv2.putText(frame, f"Reps: {counter}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    # Stage
    cv2.putText(frame, f"Stage: {stage}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2)

    # ---------- AI COACH ----------
    if stage == "DOWN":
        cv2.putText(frame, "GO UP!",
                    (300, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    3)

    if stage == "UP":
        cv2.putText(frame, "GOOD FORM!",
                    (300, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    3)

    # ---------- SHOW WINDOW ----------
    cv2.imshow("Athlete Performance AI", frame)

    # ---------- EXIT ----------
    if cv2.waitKey(1) & 0xFF == 27:
        break


# ---------- CLEANUP ----------
cap.release()
cv2.destroyAllWindows()
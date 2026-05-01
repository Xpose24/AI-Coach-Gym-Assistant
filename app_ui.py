import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
import numpy as np
import mediapipe as mp

# ---------------- CONFIG ----------------
USER_WEIGHT_KG = 70
CALORIES_PER_REP = 0.35

def calories_burned(reps):
    return reps * CALORIES_PER_REP


# ---------------- MEDIAPIPE ----------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils


# ---------------- ANGLE FUNCTION ----------------
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
              np.arctan2(a[1]-b[1], a[0]-b[0])

    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle


# ---------------- VIDEO PROCESSOR ----------------
class PoseEstimator(VideoTransformerBase):
    def __init__(self):
        self.counter = 0
        self.stage = "UP"
        self.angle = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = pose.process(rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

            knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

            ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            self.angle = calculate_angle(hip, knee, ankle)

            # -------- REP LOGIC (FIXED) --------
            if self.angle > 160:
                if self.stage == "DOWN":
                    self.counter += 1
                self.stage = "UP"

            if self.angle < 90:
                self.stage = "DOWN"

            # -------- DRAW --------
            mp_draw.draw_landmarks(
                img,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            cv2.putText(img, str(int(self.angle)),
                        tuple(np.multiply(knee, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0), 2)

        # -------- UI OVERLAY --------
        cv2.rectangle(img, (0, 0), (300, 140), (30, 30, 30), -1)

        cv2.putText(img, f"Reps: {self.counter}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        cv2.putText(img, f"Stage: {self.stage}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)

        cal = calories_burned(self.counter)

        cv2.putText(img, f"Calories: {cal:.2f}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2)

        return img


# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Athlete AI", layout="wide")

st.markdown("""
<style>
.main {
    background-color: #0f172a;
    color: white;
}
.big-card {
    background: linear-gradient(135deg, #1e293b, #020617);
    padding: 15px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 0px 15px rgba(0,255,255,0.2);
}
.metric {
    font-size: 26px;
    font-weight: bold;
}
.label {
    font-size: 16px;
    color: #94a3b8;
}
</style>
""", unsafe_allow_html=True)

# -------- HEADER --------
st.markdown("<h1 style='text-align:center;'>🏋️ Athlete Performance AI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;color:gray;'>Real-Time Squat Detection System</h4>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# -------- MOBILE FRIENDLY LAYOUT --------
col1, col2 = st.columns(2)
col3 = st.container()

reps_placeholder = col1.empty()
stage_placeholder = col2.empty()
cal_placeholder = col3.empty()

# -------- WEBCAM --------
ctx = webrtc_streamer(
    key="pose",
    video_processor_factory=PoseEstimator,
    media_stream_constraints={"video": True, "audio": False},
)

# -------- LIVE METRICS --------
if ctx.video_processor:
    reps = ctx.video_processor.counter
    stage = ctx.video_processor.stage
    calories = calories_burned(reps)

    with reps_placeholder:
        st.markdown(f"""
        <div class="big-card">
            <div class="label">Reps</div>
            <div class="metric">{reps}</div>
        </div>
        """, unsafe_allow_html=True)

    with stage_placeholder:
        color = "#22c55e" if stage == "UP" else "#facc15"
        st.markdown(f"""
        <div class="big-card">
            <div class="label">Stage</div>
            <div class="metric" style="color:{color};">{stage}</div>
        </div>
        """, unsafe_allow_html=True)

    with cal_placeholder:
        st.markdown(f"""
        <div class="big-card">
            <div class="label">Calories</div>
            <div class="metric">{calories:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
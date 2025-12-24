import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import logging
import gradio as gr
import shutil
import traceback

# --------------------------
# LOGGING SETUP
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# --------------------------
# PATHS
# --------------------------
MODEL_PATH = "models/final_raw_lstm_model.keras"
LABELS_PATH = "label/classes.npy"

# --------------------------
# LOAD MODEL & LABELS
# --------------------------
logger.info("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
logger.info(f"Model loaded | input shape: {model.input_shape}")

labels = np.load(LABELS_PATH).tolist()
logger.info(f"Loaded {len(labels)} labels")

# --------------------------
# MEDIAPIPE HOLISTIC
# --------------------------
mp_holistic = mp.solutions.holistic

# --------------------------
# KEYPOINT EXTRACTION (EXACT TRAINING LOGIC)
# --------------------------
def extract_keypoints_from_video(video_path, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    sequence = []

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            pose, lh, rh = [], [], []

            # ---- POSE (33 landmarks × 4 = 132) ----
            if results.pose_landmarks:
                for lm in results.pose_landmarks.landmark:
                    pose.extend([lm.x, lm.y, lm.z, lm.visibility])
            else:
                pose = [0] * 132

            # ---- LEFT HAND (21 × 3 = 63) ----
            if results.left_hand_landmarks:
                for lm in results.left_hand_landmarks.landmark:
                    lh.extend([lm.x, lm.y, lm.z])
            else:
                lh = [0] * 63

            # ---- RIGHT HAND (21 × 3 = 63) ----
            if results.right_hand_landmarks:
                for lm in results.right_hand_landmarks.landmark:
                    rh.extend([lm.x, lm.y, lm.z])
            else:
                rh = [0] * 63

            keypoints = np.array(pose + lh + rh)  # ✅ 258 features
            sequence.append(keypoints)

            if len(sequence) == max_frames:
                break

    cap.release()

    # ---- PAD IF SHORT VIDEO ----
    if len(sequence) < max_frames:
        sequence.extend([np.zeros(258)] * (max_frames - len(sequence)))

    return np.array(sequence)  # (30, 258)

# --------------------------
# PREDICTION FUNCTION
# --------------------------
def predict_sign(video):
    try:
        if video is None:
            return "❌ No video provided"

        # Copy to temp file
        ext = os.path.splitext(video)[1]
        temp_path = f"temp_video{ext}"
        shutil.copy(video, temp_path)

        logger.info("Extracting keypoints...")
        keypoints = extract_keypoints_from_video(temp_path)

        logger.info(f"Keypoints shape: {keypoints.shape}")

        # Add batch dimension
        input_data = np.expand_dims(keypoints, axis=0)  # (1, 30, 258)

        if input_data.shape[1:] != model.input_shape[1:]:
            return f"❌ Shape mismatch: {input_data.shape} vs {model.input_shape}"

        # Predict
        preds = model.predict(input_data, verbose=0)
        idx = np.argmax(preds[0])
        confidence = preds[0][idx]

        label = labels[idx]

        os.remove(temp_path)

        return f"✅ Predicted Sign: {label}\n🎯 Confidence: {confidence*100:.2f}%"

    except Exception as e:
        logger.error(traceback.format_exc())
        return f"❌ Error: {str(e)}"

# --------------------------
# GRADIO UI
# --------------------------
def launch_app():
    interface = gr.Interface(
        fn=predict_sign,
        inputs=gr.Video(
            label="Upload / Record ISL Video",
            sources=["upload", "webcam"],
            format="mp4"
        ),
        outputs=gr.Textbox(lines=4, label="Prediction"),
        title="🤟 ISL Recognition System",
        description=(
            "Indian Sign Language Recognition using LSTM\n\n"
            "✔ MediaPipe Holistic\n"
            "✔ 258 keypoints (Pose + Left Hand + Right Hand)\n"
            "✔ 30-frame temporal model\n\n"
            "Tip: Record directly with webcam for best results."
        ),
        cache_examples=False
    )

    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )

# --------------------------
# MAIN
# --------------------------
if __name__ == "__main__":
    logger.info("Starting ISL Recognition App...")
    launch_app()

import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import gradio as gr
import shutil
import traceback
from collections import Counter

# --------------------------
# CONFIG (TUNABLE)
# --------------------------
WINDOW_SECONDS = 5.0        # smaller window = better localization
STRIDE_SECONDS = 1.0        # overlap improves accuracy
CONF_THRESHOLD = 0.60       # reject weak predictions
MAX_FRAMES = 30

MODEL_PATH = "models/final_raw_lstm_model.keras"
LABELS_PATH = "label/classes.npy"

# --------------------------
# LOAD MODEL
# --------------------------
model = tf.keras.models.load_model(MODEL_PATH)
labels = np.load(LABELS_PATH).tolist()

# --------------------------
# MEDIAPIPE
# --------------------------
mp_holistic = mp.solutions.holistic

# --------------------------
# KEYPOINT EXTRACTION (TRAINING EXACT)
# --------------------------
def frames_to_keypoints(frames):
    sequence = []

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        for frame in frames:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            pose, lh, rh = [], [], []

            if results.pose_landmarks:
                for lm in results.pose_landmarks.landmark:
                    pose.extend([lm.x, lm.y, lm.z, lm.visibility])
            else:
                pose = [0] * 132

            if results.left_hand_landmarks:
                for lm in results.left_hand_landmarks.landmark:
                    lh.extend([lm.x, lm.y, lm.z])
            else:
                lh = [0] * 63

            if results.right_hand_landmarks:
                for lm in results.right_hand_landmarks.landmark:
                    rh.extend([lm.x, lm.y, lm.z])
            else:
                rh = [0] * 63

            sequence.append(np.array(pose + lh + rh))

            if len(sequence) == MAX_FRAMES:
                break

    if len(sequence) < MAX_FRAMES:
        sequence.extend([np.zeros(258)] * (MAX_FRAMES - len(sequence)))

    return np.array(sequence)

# --------------------------
# SLIDING WINDOW + SMOOTHING
# --------------------------
def predict_sentence(video):
    try:
        if video is None:
            return "❌ No video uploaded"

        ext = os.path.splitext(video)[1]
        temp_path = f"temp_video{ext}"
        shutil.copy(video, temp_path)

        cap = cv2.VideoCapture(temp_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps == 0 or total_frames == 0:
            return "❌ Invalid video"

        win_frames = int(WINDOW_SECONDS * fps)
        stride_frames = int(STRIDE_SECONDS * fps)

        predictions = []
        time_windows = []

        frames_buffer = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frames_buffer.append(frame)
            frame_idx += 1

            if len(frames_buffer) == win_frames:
                # sample to 30 frames
                idxs = np.linspace(0, win_frames - 1, MAX_FRAMES, dtype=int)
                sampled = [frames_buffer[i] for i in idxs]

                keypoints = frames_to_keypoints(sampled)
                input_data = np.expand_dims(keypoints, axis=0)

                preds = model.predict(input_data, verbose=0)[0]
                idx = np.argmax(preds)
                conf = preds[idx]

                start_time = (frame_idx - win_frames) / fps
                end_time = frame_idx / fps

                if conf >= CONF_THRESHOLD:
                    predictions.append(labels[idx])
                    time_windows.append((labels[idx], conf, start_time, end_time))

                # slide window
                frames_buffer = frames_buffer[stride_frames:]

        cap.release()
        os.remove(temp_path)

        # --------------------------
        # TEMPORAL SMOOTHING
        # --------------------------
        if not predictions:
            return "❌ No confident predictions"

        smoothed = []
        prev = None
        for word in predictions:
            if word != prev:
                smoothed.append(word)
                prev = word

        # Majority voting (optional extra stability)
        final_sentence = []
        buffer = []
        for w in smoothed:
            buffer.append(w)
            if len(buffer) == 3:
                final_sentence.append(Counter(buffer).most_common(1)[0][0])
                buffer = []

        if buffer:
            final_sentence.append(Counter(buffer).most_common(1)[0][0])

        # --------------------------
        # OUTPUT FORMAT
        # --------------------------
        detailed = "\n".join([
            f"🕒 {s:.1f}s–{e:.1f}s | {w} | {c*100:.1f}%"
            for w, c, s, e in time_windows
        ])

        sentence = " ".join(final_sentence)

        return (
            "📊 WINDOW PREDICTIONS:\n"
            f"{detailed}\n\n"
            "🧠 FINAL SENTENCE:\n"
            f"{sentence}"
        )

    except Exception as e:
        return f"❌ Error: {str(e)}\n{traceback.format_exc()}"

# --------------------------
# UI
# --------------------------
def launch():
    gr.Interface(
        fn=predict_sentence,
        inputs=gr.Video(sources=["upload", "webcam"], format="mp4"),
        outputs=gr.Textbox(lines=20),
        title="🤟 ISL Sentence Recognition (High Accuracy)",
        description=(
            "✔ Overlapping sliding windows\n"
            "✔ Confidence filtering\n"
            "✔ Temporal smoothing\n"
            "✔ Sentence-level output\n\n"
            "This is the highest accuracy achievable without retraining."
        ),
        cache_examples=False
    ).launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    launch()

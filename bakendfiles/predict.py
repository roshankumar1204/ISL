import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# ------------------------------
# 🔹 CONFIGURATION
# ------------------------------
MODEL_PATH = "../models/isl_sign_model.h5"  # Path to your trained model
ACTIONS = np.array(['Good Morning', 'Give', 'Crocodile', 'Maybe', 'Knife'])
SEQUENCE_LENGTH = 30

# ------------------------------
# 🔹 LOAD MODEL
# ------------------------------
print("🔁 Loading model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# ------------------------------
# 🔹 Initialize MediaPipe
# ------------------------------
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


# ------------------------------
# 🔹 Function to extract keypoints
# ------------------------------
def extract_keypoints(results):
    # Pose (33), Left hand (21), Right hand (21)
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, lh, rh])


# ------------------------------
# 🔹 Preprocess Video → Keypoints Sequence
# ------------------------------
def video_to_sequence(video_path):
    sequence = []
    cap = cv2.VideoCapture(video_path)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
    cap.release()

    # Pad or trim to fixed SEQUENCE_LENGTH
    if len(sequence) < SEQUENCE_LENGTH:
        # Pad with zeros
        for _ in range(SEQUENCE_LENGTH - len(sequence)):
            sequence.append(np.zeros(sequence[0].shape))
    else:
        sequence = sequence[:SEQUENCE_LENGTH]

    return np.array(sequence)


# ------------------------------
# 🔹 Predict Function
# ------------------------------
def predict_sign(video_path):
    sequence = video_to_sequence(video_path)
    input_data = np.expand_dims(sequence, axis=0)  # Shape: (1, 30, n_features)
    predictions = model.predict(input_data)[0]
    predicted_index = np.argmax(predictions)
    predicted_label = ACTIONS[predicted_index]
    confidence = predictions[predicted_index] * 100

    print(f"\n🟩 Predicted Sign → \"{predicted_label}\" ({confidence:.2f}%)")
    print("\n📊 Confidence Scores:")
    for i, action in enumerate(ACTIONS):
        print(f"{action:<15}: {predictions[i]*100:.2f}%")

    return predicted_label, confidence


# ------------------------------
# 🔹 Run Example
# ------------------------------
if __name__ == "__main__":
    video_path = input("🎥 Enter path of video to test: ").strip()
    predict_sign(video_path)

import os
import cv2
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic


# -------------------------------------
#   KEYPOINT EXTRACTION FUNCTIONS
# -------------------------------------

def extract_keypoints_from_frame(results):
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

    return np.array(pose + lh + rh)


def video_to_keypoints(video_path, max_frames=30):
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

            keypoints = extract_keypoints_from_frame(results)
            sequence.append(keypoints)

            if len(sequence) == max_frames:
                break

    cap.release()

    # pad if video too short
    if len(sequence) < max_frames:
        sequence.extend([np.zeros(258)] * (max_frames - len(sequence)))

    return np.array(sequence)



# -------------------------------------
#   DATASET CREATION (TOP 40 PER CLASS)
# -------------------------------------

DATA_PATH = "isl_data/Video_Dataset/Video_Dataset"
actions = os.listdir(DATA_PATH)

X, y = [], []

print("\n📂 Starting dataset creation...\n")

for idx, action in enumerate(actions):
    folder = os.path.join(DATA_PATH, action)
    videos = sorted(os.listdir(folder))

    top_videos = videos[:55]  # take only first 55 videos

    print(f"▶ Processing class: {action} ({len(top_videos)} videos)")

    for i, video in enumerate(top_videos):
        video_path = os.path.join(folder, video)

        sequence = video_to_keypoints(video_path)

        X.append(sequence)
        y.append(idx)

        print(f"   ✅ [{i+1}/{len(top_videos)}] {video} → processed")

print("\n🎯 Dataset creation complete.\n")

X = np.array(X)   # (#samples, 30, 258)
y = np.array(y)   # (#samples,)

print("🔎 Final dataset shapes:")
print("   X:", X.shape)
print("   y:", y.shape)


# -------------------------------------
#   SAVE PREPROCESSED DATA
# -------------------------------------

SAVE_DIR = "preprocessed_data"
os.makedirs(SAVE_DIR, exist_ok=True)

np.save(os.path.join(SAVE_DIR, "X_keypoints.npy"), X)
np.save(os.path.join(SAVE_DIR, "y_labels.npy"), y)

print("\n💾 Saved preprocessed data:")
print(f"   {SAVE_DIR}/X_keypoints.npy")
print(f"   {SAVE_DIR}/y_labels.npy")
print("\n✅ Done!")

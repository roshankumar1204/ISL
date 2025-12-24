import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import gradio as gr
import traceback
import google.generativeai as genai

from norm_videopath import normalize_video_path
from hindi_tts import text_to_speech_hindi, text_to_speech_from_sentence


# ENV SAFETY (WINDOWS)

os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["GLOG_minloglevel"] = "2"  # Suppress MediaPipe logs


# CONFIG

MAX_FRAMES = 30
MODEL_PATH = "models/final_raw_lstm_model.keras"
LABELS_PATH = "label/classes.npy"


# LOAD MODEL & LABELS

model = tf.keras.models.load_model(MODEL_PATH)
labels = np.load(LABELS_PATH).tolist()


# GEMINI SETUP

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

gemini = genai.GenerativeModel(
    model_name="models/gemini-flash-latest"
)


# MEDIAPIPE SOLUTIONS

mp_holistic = mp.solutions.holistic


# VIDEO → KEYPOINTS

def video_to_keypoints(video_path):
    """
    Extract keypoints from video using MediaPipe Holistic.
    Creates a NEW holistic instance per video to avoid state issues.
    """
    # Normalize path for Windows
    video_path = os.path.abspath(video_path).replace('\\', '/')
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    sequence = []
    
    # Create holistic instance per video (important for thread safety)
    try:
        holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    except Exception as e:
        cap.release()
        raise RuntimeError(f"Failed to initialize MediaPipe Holistic: {e}")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Make image writeable False to improve performance
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True

            pose, lh, rh = [], [], []

            # Pose (33 landmarks × 4 values = 132)
            if results.pose_landmarks:
                for lm in results.pose_landmarks.landmark:
                    pose.extend([lm.x, lm.y, lm.z, lm.visibility])
            else:
                pose = [0.0] * 132

            # Left hand (21 landmarks × 3 values = 63)
            if results.left_hand_landmarks:
                for lm in results.left_hand_landmarks.landmark:
                    lh.extend([lm.x, lm.y, lm.z])
            else:
                lh = [0.0] * 63

            # Right hand (21 landmarks × 3 values = 63)
            if results.right_hand_landmarks:
                for lm in results.right_hand_landmarks.landmark:
                    rh.extend([lm.x, lm.y, lm.z])
            else:
                rh = [0.0] * 63

            # Combine all features (132 + 63 + 63 = 258)
            frame_keypoints = pose + lh + rh
            sequence.append(np.array(frame_keypoints, dtype=np.float32))

            if len(sequence) >= MAX_FRAMES:
                break
    
    finally:
        cap.release()
        holistic.close()

    # Pad or truncate to MAX_FRAMES
    if len(sequence) < MAX_FRAMES:
        padding = [np.zeros(258, dtype=np.float32)] * (MAX_FRAMES - len(sequence))
        sequence.extend(padding)
    elif len(sequence) > MAX_FRAMES:
        sequence = sequence[:MAX_FRAMES]

    return np.array(sequence, dtype=np.float32)


# PREDICT SINGLE WORD (WEBCAM)

def predict_word(video, sentence):
    """Predict a single word from webcam video"""
    try:
        if video is None:
            return sentence, "❌ No video recorded"

        safe_path = normalize_video_path(video)
        
        # Check if file exists and is valid
        if not os.path.exists(safe_path):
            return sentence, "❌ Video file not found"
        
        keypoints = video_to_keypoints(safe_path)
        
        # Clean up video file
        try:
            os.remove(safe_path)
        except:
            pass

        # Predict
        inp = np.expand_dims(keypoints, axis=0)
        preds = model.predict(inp, verbose=0)[0]

        idx = np.argmax(preds)
        conf = preds[idx]
        word = labels[idx]

        # Add to sentence
        sentence.append(word)

        return (
            sentence,
            f"✅ Added: {word} ({conf*100:.2f}%)\n\n🧠 Current words:\n{' '.join(sentence)}"
        )

    except Exception as e:
        return sentence, f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"


# PREDICT MULTIPLE VIDEOS

def predict_multiple_videos(videos, sentence):
    """Predict words from multiple uploaded videos"""
    try:
        if not videos:
            return sentence, "❌ No videos uploaded"

        logs = []

        for v in videos:
            safe_path = normalize_video_path(v)
            
            if not os.path.exists(safe_path):
                logs.append(f"⚠️ Skipped: {os.path.basename(safe_path)} (not found)")
                continue
            
            keypoints = video_to_keypoints(safe_path)
            
            # Clean up
            try:
                os.remove(safe_path)
            except:
                pass

            # Predict
            inp = np.expand_dims(keypoints, axis=0)
            preds = model.predict(inp, verbose=0)[0]

            idx = np.argmax(preds)
            conf = preds[idx]
            word = labels[idx]

            sentence.append(word)
            logs.append(f"✅ {word} ({conf*100:.2f}%)")

        return (
            sentence,
            "📊 Added Words:\n"
            + "\n".join(logs)
            + "\n\n🧠 Current Sentence:\n"
            + " ".join(sentence)
        )

    except Exception as e:
        return sentence, f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"


# GEMINI SENTENCE ANALYSIS WITH TTS

def analyze_with_gemini(sentence_words):
    """Generate natural Hindi sentence from ISL words using Gemini and convert to speech"""
    try:
        if not sentence_words:
            return "❌ No words to analyze. Please add words first.", None

        words_str = " ".join(sentence_words)
        
        prompt = f"""You are a Hindi sentence generator for Indian Sign Language (ISL).

Input ISL words (in order): {words_str}

Task:
1. These words represent signs performed in ISL
2. Translate each word to Hindi if needed
3. Arrange them in proper Hindi grammar order (Subject-Object-Verb typical)
4. You may add necessary auxiliary verbs (है, था, हूँ, etc.) or postpositions (को, से, में, etc.) for naturalness
5. Do NOT add new content words beyond the input
6. Output ONLY the final Hindi sentence

Output the Hindi sentence:"""

        response = gemini.generate_content(prompt)
        hindi_sentence = response.text.strip()
        
        # Remove any markdown formatting
        hindi_sentence = hindi_sentence.replace("**", "").replace("*", "")
        
        # Generate speech from Hindi sentence
        audio_path, tts_msg = text_to_speech_hindi(hindi_sentence)
        
        result_text = f"🧠 Natural Hindi Sentence:\n\n{hindi_sentence}\n\n📝 Original words: {words_str}\n\n{tts_msg}"
        
        return result_text, audio_path
    
    except Exception as e:
        return f"❌ Error: {str(e)}\n\nPlease check your API key and internet connection.", None


# RESET

def reset_sentence():
    """Clear the current sentence"""
    return [], "🔄 Sentence cleared. Ready for new words.", "", None


# UI

def launch():
    with gr.Blocks(title="ISL Transalation") as demo:
        gr.Markdown("# 🤟 ISL Transalation")

        sentence_state = gr.State([])

        # Mode selector
        mode = gr.Radio(
            choices=["📹 Webcam", "📁 Upload Files"],
            value="📹 Webcam",
            label="Choose Input Method"
        )

        # Webcam section
        with gr.Group(visible=True) as webcam_section:
            cam = gr.Video(sources=["webcam"], format="mp4")
            webcam_btn = gr.Button("Add Sign", variant="primary", size="lg")

        # Upload section
        with gr.Group(visible=False) as upload_section:
            files = gr.File(file_types=["video"], file_count="multiple")
            upload_btn = gr.Button("Add Signs", variant="primary", size="lg")

        # Result output (define before using)
        result_out = gr.Textbox(label="Result", lines=3)

        gr.Markdown("---")
        
        # Action buttons and output (define before using)
        with gr.Row():
            hindi_btn = gr.Button("🧠 Generate Hindi", variant="secondary", size="lg")
            clear_btn = gr.Button("🔄 Clear", size="lg")
        
        analyze_out = gr.Textbox(label="Hindi Sentence", lines=3)
        audio_out = gr.Audio(label="🔊 Hindi Speech", type="filepath", autoplay=True)

        # Now connect all buttons
        webcam_btn.click(
            predict_word,
            inputs=[cam, sentence_state],
            outputs=[sentence_state, result_out]
        )

        upload_btn.click(
            predict_multiple_videos,
            inputs=[files, sentence_state],
            outputs=[sentence_state, result_out]
        )

        hindi_btn.click(
            analyze_with_gemini,
            inputs=[sentence_state],
            outputs=[analyze_out, audio_out]
        )

        clear_btn.click(
            reset_sentence,
            outputs=[sentence_state, result_out, analyze_out, audio_out]
        )

        # Toggle visibility based on mode
        def update_mode(choice):
            if choice == "📹 Webcam":
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)

        mode.change(
            update_mode,
            inputs=[mode],
            outputs=[webcam_section, upload_section]
        )

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )


# MAIN

if __name__ == "__main__":
    launch()
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import logging
import gradio as gr
import os
import shutil
import traceback
from pathlib import Path

# --------------------------
# ENHANCED LOGGING SETUP
# --------------------------
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more detailed logs
    format="%(asctime)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s"
)
logger = logging.getLogger(__name__)

# --------------------------
# LOAD MODEL + LABELS
# --------------------------
MODEL_PATH = "models/final_raw_lstm_model.keras"
LABELS_PATH = "label/classes.npy"

logger.info("="*50)
logger.info("INITIALIZING ISL RECOGNITION SYSTEM")
logger.info("="*50)

try:
    logger.info(f"Loading model from: {MODEL_PATH}")
    logger.info(f"Model file exists: {os.path.exists(MODEL_PATH)}")
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info(f"Model loaded successfully")
    logger.info(f"Model input shape: {model.input_shape}")
    logger.info(f"Model output shape: {model.output_shape}")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    logger.error(traceback.format_exc())
    raise

try:
    logger.info(f"Loading labels from: {LABELS_PATH}")
    logger.info(f"Labels file exists: {os.path.exists(LABELS_PATH)}")
    labels = np.load(LABELS_PATH)
    labels = labels.tolist()
    logger.info(f"Loaded {len(labels)} labels: {labels}")
except Exception as e:
    logger.error(f"Failed to load labels: {str(e)}")
    logger.error(traceback.format_exc())
    raise


# --------------------------
# EXTRACT KEYPOINTS FUNCTION
# --------------------------
mp_hands = mp.solutions.hands

def extract_keypoints_from_video(video_path):
    """Extract MediaPipe hand + pose keypoints from video to match model's 258 features."""
    logger.info("-"*50)
    logger.info(f"EXTRACTING KEYPOINTS FROM: {video_path}")
    logger.info("-"*50)
    
    if not os.path.exists(video_path):
        logger.error(f"Video file does not exist: {video_path}")
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    file_size = os.path.getsize(video_path)
    logger.info(f"Video file size: {file_size / (1024*1024):.2f} MB")

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        raise Exception(f"OpenCV could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Video properties: {width}x{height}, {fps:.2f} FPS, {frame_count_total} frames")
    
    all_keypoints = []
    frame_count = 0
    hands_detected_count = 0

    # Initialize both hands and pose
    mp_pose = mp.solutions.pose
    
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands, mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.debug(f"End of video reached at frame {frame_count}")
                break

            frame_count += 1
            
            if frame_count % 30 == 0:  # Log every 30 frames
                logger.debug(f"Processing frame {frame_count}/{frame_count_total}")

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process hands
            hand_result = hands.process(frame_rgb)
            # Process pose
            pose_result = pose.process(frame_rgb)

            # Hands: 21 landmarks × 2 hands × 3 coords = 126
            hand_keypoints = np.zeros(126)
            
            if hand_result.multi_hand_landmarks:
                hands_detected_count += 1
                num_hands = len(hand_result.multi_hand_landmarks)
                
                for hand_idx, hand_landmarks in enumerate(hand_result.multi_hand_landmarks):
                    if hand_idx >= 2:  # Only process first 2 hands
                        break
                    hkp = []
                    for lm in hand_landmarks.landmark:
                        hkp.extend([lm.x, lm.y, lm.z])
                    
                    # Place in correct position (0-63 for first hand, 63-126 for second)
                    start_idx = hand_idx * 63
                    hand_keypoints[start_idx:start_idx + len(hkp)] = hkp
                    
                    if frame_count % 30 == 0:
                        logger.debug(f"Frame {frame_count}: Hand {hand_idx} - {len(hkp)} coords at index {start_idx}")
            
            # Pose: 33 landmarks × 3 coords = 99, but we need to reach 258 total
            # 126 (hands) + 132 (pose upper body landmarks) = 258
            pose_keypoints = np.zeros(132)
            
            if pose_result.pose_landmarks:
                # Extract upper body pose landmarks (shoulders, elbows, wrists, etc.)
                # Using first 44 landmarks × 3 = 132
                pkp = []
                for i, lm in enumerate(pose_result.pose_landmarks.landmark):
                    if i >= 44:  # Limit to 44 landmarks
                        break
                    pkp.extend([lm.x, lm.y, lm.z])
                pose_keypoints[:len(pkp)] = pkp
                
                if frame_count % 30 == 0:
                    logger.debug(f"Frame {frame_count}: Pose - {len(pkp)} coords extracted")
            
            # Combine: 126 (hands) + 132 (pose) = 258 features
            combined_keypoints = np.concatenate([hand_keypoints, pose_keypoints])
            
            if frame_count == 1:
                logger.info(f"Combined keypoints shape per frame: {combined_keypoints.shape}")
            
            all_keypoints.append(combined_keypoints)

    cap.release()
    
    all_keypoints = np.array(all_keypoints)
    logger.info(f"Keypoint extraction complete:")
    logger.info(f"  - Total frames processed: {frame_count}")
    logger.info(f"  - Frames with hands detected: {hands_detected_count} ({hands_detected_count/frame_count*100:.1f}%)")
    logger.info(f"  - Keypoints array shape: {all_keypoints.shape}")
    logger.info(f"  - Features per frame: {all_keypoints.shape[1]} (expected: 258)")
    
    if all_keypoints.shape[1] != 258:
        logger.error(f"Feature mismatch! Got {all_keypoints.shape[1]}, expected 258")
    
    return all_keypoints


# --------------------------
# PREDICT FUNCTION
# --------------------------
def predict_sign(video):
    """Main prediction function with comprehensive logging."""
    logger.info("="*50)
    logger.info("PREDICTION STARTED")
    logger.info("="*50)
    
    try:
        # Step 1: Validate input
        logger.info(f"Step 1: Validating input")
        logger.info(f"  - Input type: {type(video)}")
        logger.info(f"  - Input value: {video}")
        
        if video is None:
            logger.error("No video provided (input is None)")
            return "ERROR: No video uploaded"
        
        # Step 2: Validate and prepare video file
        logger.info(f"Step 2: Validating and preparing video file")
        original_path = video
        temp_path = "temp_input_video.mp4"
        
        # Remove old temp file
        if os.path.exists(temp_path):
            logger.debug(f"Removing existing temp file: {temp_path}")
            os.remove(temp_path)
        
        logger.info(f"  - Source: {original_path}")
        logger.info(f"  - Source file exists: {os.path.exists(original_path)}")
        
        if not os.path.exists(original_path):
            logger.error(f"Source video file does not exist: {original_path}")
            return "ERROR: Video file not found. Please try uploading again."
        
        original_size = os.path.getsize(original_path)
        logger.info(f"  - Original file size: {original_size / (1024*1024):.2f} MB")
        
        # Get file extension
        file_ext = os.path.splitext(original_path)[1].lower()
        logger.info(f"  - File extension: {file_ext}")
        
        # Copy to temp location with proper extension
        if file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            temp_path = f"temp_input_video{file_ext}"
        
        logger.info(f"  - Copying to: {temp_path}")
        try:
            shutil.copy(original_path, temp_path)
            logger.info(f"  - Copy successful")
        except Exception as copy_error:
            logger.error(f"Failed to copy file: {str(copy_error)}")
            return f"ERROR: Could not process video file: {str(copy_error)}"
        
        # Verify the video can be opened
        logger.info(f"  - Verifying video can be opened by OpenCV...")
        test_cap = cv2.VideoCapture(temp_path)
        can_open = test_cap.isOpened()
        
        if can_open:
            # Get video info
            fps = test_cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(f"  - Video opened successfully: {width}x{height}, {fps:.2f} FPS, {frame_count} frames")
        else:
            logger.error(f"  - OpenCV cannot open the video file")
            test_cap.release()
            return f"ERROR: Video format not supported. Please try:\n1. Recording directly with webcam\n2. Converting video to MP4 format\n3. Using a different video file"
        
        test_cap.release()
        logger.info(f"  - Video validation complete")
        
        # Step 3: Extract keypoints
        logger.info(f"Step 3: Extracting keypoints using MediaPipe")
        keypoints = extract_keypoints_from_video(temp_path)
        
        if len(keypoints) == 0:
            logger.warning("No keypoints extracted from video")
            return "ERROR: No frames could be processed"
        
        # Step 4: Prepare input for model
        logger.info(f"Step 4: Preparing model input")
        logger.info(f"  - Keypoints shape before processing: {keypoints.shape}")
        
        # Model expects: (batch, 30 frames, 258 features)
        required_frames = 30
        current_frames = keypoints.shape[0]
        
        logger.info(f"  - Current frames: {current_frames}, Required: {required_frames}")
        
        if current_frames < required_frames:
            # Pad with zeros if not enough frames
            logger.warning(f"Video has only {current_frames} frames, padding to {required_frames}")
            padding = np.zeros((required_frames - current_frames, 258))
            keypoints = np.vstack([keypoints, padding])
            logger.info(f"  - After padding: {keypoints.shape}")
        elif current_frames > required_frames:
            # Sample frames evenly if too many
            logger.info(f"Video has {current_frames} frames, sampling {required_frames} frames")
            indices = np.linspace(0, current_frames - 1, required_frames, dtype=int)
            keypoints = keypoints[indices]
            logger.info(f"  - After sampling: {keypoints.shape}")
        
        # Add batch dimension
        keypoints_input = np.expand_dims(keypoints, axis=0)
        logger.info(f"  - Final shape after batch expansion: {keypoints_input.shape}")
        logger.info(f"  - Expected model input shape: {model.input_shape}")
        
        # Check shape compatibility
        if keypoints_input.shape[1:] != model.input_shape[1:]:
            logger.error(f"Shape mismatch! Input: {keypoints_input.shape}, Expected: {model.input_shape}")
            return f"ERROR: Shape mismatch - got {keypoints_input.shape}, expected {model.input_shape}"
        
        # Step 5: Make prediction
        logger.info(f"Step 5: Running model prediction")
        predictions = model.predict(keypoints_input, verbose=0)
        logger.info(f"  - Prediction shape: {predictions.shape}")
        logger.info(f"  - Prediction values (first 5): {predictions[0][:5]}")
        
        # Step 6: Get result
        logger.info(f"Step 6: Processing prediction results")
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx]
        
        logger.info(f"  - Predicted index: {predicted_idx}")
        logger.info(f"  - Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        
        if predicted_idx >= len(labels):
            logger.error(f"Predicted index {predicted_idx} out of range (max: {len(labels)-1})")
            return f"ERROR: Invalid prediction index"
        
        predicted_label = labels[predicted_idx]
        logger.info(f"  - Predicted label: {predicted_label}")
        
        # Step 7: Cleanup
        logger.info(f"Step 7: Cleanup")
        try:
            os.remove(temp_path)
            logger.info(f"  - Removed temp file: {temp_path}")
        except Exception as cleanup_error:
            logger.warning(f"  - Could not remove temp file: {cleanup_error}")
        
        # Final result
        result = f"Predicted Sign: {predicted_label}\nConfidence: {confidence*100:.2f}%"
        logger.info("="*50)
        logger.info(f"PREDICTION SUCCESSFUL: {predicted_label}")
        logger.info("="*50)
        
        return result
        
    except Exception as e:
        logger.error("="*50)
        logger.error("PREDICTION FAILED")
        logger.error("="*50)
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        
        return f"ERROR: {type(e).__name__}: {str(e)}"


# --------------------------
# GRADIO UI
# --------------------------
def launch_app():
    """Launch Gradio interface."""
    logger.info("Launching Gradio interface...")
    
    interface = gr.Interface(
        fn=predict_sign,
        inputs=gr.Video(
            label="Upload ISL Video",
            sources=["upload", "webcam"],  # Allow both upload and recording
            format="mp4"  # Specify output format
        ),
        outputs=gr.Textbox(label="Prediction Output", lines=5),
        title="🤟 ISL Recognition System",
        description=(
            "Upload or record an Indian Sign Language video for gesture classification.\n\n"
            "**Supported formats:** MP4, AVI, MOV\n"
            "**Recommended:** Record directly using webcam for best compatibility\n"
            "**Model:** LSTM trained on 61 ISL gestures with hand + pose keypoints"
        ),
        examples=None,
        cache_examples=False
    )

    logger.info("Starting server on 0.0.0.0:7860")
    interface.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        show_error=True  # Show errors in the interface
    )


if __name__ == "__main__":
    logger.info("Starting ISL Recognition Application...")
    try:
        launch_app()
    except Exception as e:
        logger.critical(f"Application failed to start: {str(e)}")
        logger.critical(traceback.format_exc())
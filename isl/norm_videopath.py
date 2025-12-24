import os
import shutil
import tempfile

def normalize_video_path(video):
    """
  Copy to a simple temp path with forward slashes
    """
    # Get source path
    if isinstance(video, str):
        src_path = video
    else:
        src_path = video.name if hasattr(video, 'name') else str(video)
    
    # Validate source exists
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Video file not found: {src_path}")
    
    # Get file extension
    ext = os.path.splitext(src_path)[1]
    if not ext:
        ext = '.mp4'  # default extension
    
    # Create temp file with simple name
    tmp_dir = tempfile.gettempdir()
    tmp_name = f"mediapipe_video_{os.getpid()}_{id(video)}{ext}"
    tmp_path = os.path.join(tmp_dir, tmp_name)
    
    # Copy file
    shutil.copy2(src_path, tmp_path)
    
    # Convert to forward slashes (MediaPipe compatible)
    normalized_path = tmp_path.replace('\\', '/')
    
    return normalized_path
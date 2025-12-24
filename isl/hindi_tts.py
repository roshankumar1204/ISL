import os
import tempfile
from gtts import gTTS
import traceback

def text_to_speech_hindi(text):
    """
    Convert Hindi text to speech using Google TTS
    
    Args:
        text (str): Hindi text to convert to speech
        
    Returns:
        tuple: (audio_file_path, status_message)
    """
    try:
        if not text or text.strip() == "":
            return None, "❌ No text to convert"
        
        # Clean the text
        clean_text = text.strip()
        
        # Create TTS object with Hindi language
        tts = gTTS(text=clean_text, lang='hi', slow=False)
        
        # Create temporary file for audio
        temp_dir = tempfile.gettempdir()
        audio_file = os.path.join(temp_dir, f"hindi_speech_{os.getpid()}.mp3")
        
        # Save audio file
        tts.save(audio_file)
        
        return audio_file, f"✅ Audio generated successfully"
        
    except Exception as e:
        return None, f"❌ TTS Error: {str(e)}\n{traceback.format_exc()}"


def text_to_speech_from_sentence(sentence_words):
    """
    Convert sentence words list to Hindi speech
    
    Args:
        sentence_words (list): List of words from ISL predictions
        
    Returns:
        tuple: (audio_file_path, status_message)
    """
    try:
        if not sentence_words or len(sentence_words) == 0:
            return None, "❌ No words to convert. Please add signs first."
        
        # Join words into sentence
        text = " ".join(sentence_words)
        
        # Generate speech
        return text_to_speech_hindi(text)
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


# Test function
if __name__ == "__main__":
    # Test 1: Simple Hindi text
    print("Test 1: Simple Hindi text")
    audio_path, msg = text_to_speech_hindi("नमस्ते, मैं आपकी मदद कर सकता हूं")
    print(msg)
    if audio_path:
        print(f"Audio saved at: {audio_path}")
    
    # Test 2: From sentence words
    print("\nTest 2: From sentence words")
    words = ["नमस्ते", "मैं", "खाना", "खाता", "हूं"]
    audio_path, msg = text_to_speech_from_sentence(words)
    print(msg)
    if audio_path:
        print(f"Audio saved at: {audio_path}")
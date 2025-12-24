from gtts import gTTS
import os

text = "नमस्ते, मैं आपकी मदद कर सकता हूं"
tts = gTTS(text=text, lang='hi')
tts.save("output.mp3")
os.system("output.mp3")  # Play on Windows
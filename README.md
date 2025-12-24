# 🤟 ISL Translator – Indian Sign Language to Hindi

An **end-to-end multimodal artificial intelligence system** designed to translate **Indian Sign Language (ISL)** gestures from video input into grammatically correct **Hindi text and speech** through **deep learning and generative AI techniques**.

---

## 🚀 Overview

This project bridges the communication gap between **ISL users and Hindi speakers** by combining:
- **Computer Vision** for gesture understanding
- **Deep Learning (BiLSTM)** for temporal modeling
- **LLM-based reasoning (Gemini)** for natural language generation
- **Text-to-Speech** for accessible audio output

The system is designed to be **modular, scalable, and production-ready**.

---

## 🧠 System Architecture

Video Input (Webcam / Upload)
↓
MediaPipe Holistic (Pose + Hand Keypoints)
↓
Sequence Normalization (30 frames)
↓
BiLSTM Gesture Classification
↓
ISL Word Predictions
↓
Gemini LLM (Hindi Grammar & Ordering)
↓
Hindi Text + Speech Output





---

## 🔑 Key Features

- 🎥 **Dynamic ISL gesture recognition** from video (not static images)
- 🧠 **Bidirectional LSTM** for temporal motion understanding
- ✨ **Gemini LLM integration** for grammar-aware Hindi sentence generation
- 🔊 **Hindi Text-to-Speech** for real-world accessibility
- ⚙️ **Robust preprocessing** handling variable FPS and video lengths
- 🧩 **Modular pipeline** suitable for API-based deployment

---

## 🛠️ Tech Stack

- **Python**
- **TensorFlow / Keras**
- **MediaPipe Holistic**
- **OpenCV**
- **NumPy**
- **Google Gemini API**
- **Gradio (UI)**

---

## 📐 Model Details

- **Input Shape:** `(30 frames × 258 features)`
  - Pose: 132
  - Left Hand: 63
  - Right Hand: 63
- **Architecture:**
  - Bidirectional LSTM (256 units)
  - Bidirectional LSTM (128 units)
  - Dense + Softmax
- **Loss:** Sparse Categorical Crossentropy  
- **Optimizer:** Adam

---

## ▶️ How It Works

1. User records or uploads an ISL gesture video  
2. MediaPipe extracts pose and hand keypoints per frame  
3. Video is normalized to a fixed-length sequence  
4. BiLSTM model predicts the corresponding ISL word  
5. Gemini LLM restructures predicted words into natural Hindi  
6. Hindi sentence is displayed and spoken aloud  

---

## ⚠️ Current Limitations

- Limited ISL vocabulary
- Optimized for **word-level** gestures (not continuous signing)
- Performance may vary in poor lighting conditions

---

## 🔮 Future Enhancements

- Sentence-level and continuous sign recognition
- Regional language support (Kannada, Tamil, etc.)
- FastAPI-based microservices deployment
- Mobile application support
- Incremental learning with verified data

---

## 📌 Why This Project Matters

- ISL is **underrepresented** compared to ASL in AI research
- Promotes **accessibility and inclusion**
- Demonstrates **real-world application of Generative AI + ML**
- Designed with **production-readiness** in mind

---

## Demo (screenshots of training data and working)

<img width="1538" height="691" alt="trainning " src="https://github.com/user-attachments/assets/d9344a3d-0764-47e4-95bf-4758a1fcf82e" />
<img width="1920" height="1080" alt="working 4 (1)" src="https://github.com/user-attachments/assets/c99c65cc-9220-4e6d-9a24-526937dc4f2f" />

---

## 👤 Author

**Roshan Kumar**  
**Aman jaiswal**  
**Ravi pratap singh**  
**Nikhil Sahu**  

---

## 📄 License

This project is for educational and research purposes.

This project is for educational and research purposes.

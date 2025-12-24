# 🎭 Emotion Recognition System

## 🚀 Introduction
Welcome to the **Emotion Recognition System**! This application uses AI to detect emotions based on **facial expressions** and **speech tone**. It leverages deep learning techniques to provide accurate emotion predictions from either uploaded images (for face-based emotion recognition) or audio files (for speech-based emotion recognition).

### 💡 Features:
- **Face-only Mode**: Detects emotions based on facial expressions.
- **Multimodal Mode**: Combines both facial expressions and speech tone for more accurate results.

---

## 🛠️ Requirements

Before running the app, you need to install the following dependencies:

- **Python 3.7+**
- **Streamlit**: For running the frontend (Web UI).
- **TensorFlow**: For running the pre-trained emotion models.
- **Librosa**: To extract audio features (MFCC).
- **OpenCV**: To process the images.

You can install all required dependencies with the following command:

```bash
pip install -r requirements.txt
```

### Example `requirements.txt`:

```txt
streamlit==1.28.0
tensorflow==2.15.0
protobuf==3.20.3
numpy==1.24.3
opencv-python==4.8.1.78
librosa==0.10.1
Pillow==10.1.0
scikit-learn==1.3.2
pandas==2.1.3
matplotlib==3.8.2
seaborn==0.13.0
resampy==0.4.2
soundfile==0.12.1
audioread==3.0.1
numba==0.58.1
```

---

## 📝 Setup and Run

### 1. Clone the repository:

```bash
git https://github.com/Lamis4/multimodal-emotion-recognition.git
cd emotion-recognition-app
```

### 2. Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Download the Pretrained Models:

This app requires **Facial Emotion Recognition (FER)** and **Speech Emotion Recognition (SER)** models. Ensure that the files `fer_model.h5` and `ser_model.h5` are located in the project directory.

If you don't have the models yet, you can either train them yourself or find pre-trained models online.

### 4. Run the app:

Once the requirements are installed and the models are downloaded, run the app with:

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501/` in your browser.

---

## 💻 How to Use the App

### Face-only Mode:
Upload a clear image of a face (preferably well-lit) to get an emotion prediction based on facial expressions.

### Multimodal Mode:
Upload both an image and an audio file. The app will analyze both and give you the final emotion prediction based on both inputs.

### Supported File Formats:
- **Images**: `.jpg`, `.jpeg`, `.png`
- **Audio**: `.wav`, `.mp3`

---

## 🔧 Troubleshooting

### TensorFlow Import Error
If you get an error related to TensorFlow or protobuf, you can fix it by running:

```bash
pip uninstall protobuf
pip install protobuf==3.20.3
```

### Missing Audio Libraries
If you get an error about `resampy` or other audio libraries:

```bash
pip install resampy soundfile audioread
```

### Low Confidence Predictions
If you get warnings for low confidence, try uploading:
- Clearer images with good lighting
- Higher-quality audio files
- Audio files that are 2-5 seconds long

---

## 🧠 Model Information

### FER Model (Facial Emotion Recognition)
This model predicts emotions based on facial expressions using deep neural networks.

**Supported Emotions:**
- `0`: Angry 😠
- `1`: Disgust 🤢
- `2`: Fear 😨
- `3`: Happy 😊
- `4`: Sad 😢
- `5`: Surprise 😲
- `6`: Neutral 😐

### SER Model (Speech Emotion Recognition)
This model analyzes audio to predict emotions based on speech tone using extracted MFCC features.

**Supported Emotions:**
- `01`: Neutral 😐
- `02`: Calm 😌
- `03`: Happy 😊
- `04`: Sad 😢
- `05`: Angry 😠
- `06`: Fearful 😰
- `07`: Disgust 🤢
- `08`: Surprised 😲

---

## 📊 Project Structure

```
emotion-recognition-app/
│
├── app.py                  # Main Streamlit application
├── fer_model.h5           # Facial Emotion Recognition model
├── ser_model.h5           # Speech Emotion Recognition model
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

---

## 🎯 Performance Tips

- **For Face Recognition**: Use clear, front-facing images with good lighting
- **For Speech Recognition**: Use clear audio recordings without background noise
- **For Best Results**: Use both modalities in Multimodal Mode for improved accuracy

---



## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Lamis4/).

---

## 👨‍💻 Authors

- **Afaf saed altalhi** 
- **Lamis Mohamed alsaedi**
- **Altaf Saud Alsulami**
- **Manal Saud Alsulami** 
- **Renad wael alhalabi** 
---

## 🙏 Acknowledgments

- FER2013 dataset for facial emotion recognition
- RAVDESS dataset for speech emotion recognition
- TensorFlow and Keras for deep learning frameworks
- Streamlit for the amazing web framework

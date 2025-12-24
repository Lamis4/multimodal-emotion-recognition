import streamlit as st
import numpy as np
import cv2
import librosa
from PIL import Image
import io

# Try to import TensorFlow with error handling
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError as e:
    st.error(f"⚠️ TensorFlow import error: {str(e)}")
    st.info("Please fix the protobuf version issue by running: `pip install protobuf==3.20.3`")
    TF_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Emotion Recognition System",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-weight: 600;
        font-size: 16px;
    }
    .upload-box {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9fa;
    }
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    h1 {
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    h2 {
        color: #334155;
    }
    h3 {
        color: #475569;
    }
    </style>
""", unsafe_allow_html=True)

# Constants
FER_EMOTION_LABELS = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
RAVDESS_EMOTION_LABELS = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}
IMG_SIZE = 48
MFCC_COUNT = 40
MAX_PAD_LEN = 174
SAMPLE_RATE = 22050

# Emotion to emoji mapping
EMOTION_EMOJIS = {
    'angry': '😠', 'disgust': '🤢', 'fear': '😨', 'happy': '😊',
    'sad': '😢', 'surprise': '😲', 'neutral': '😐', 'calm': '😌',
    'fearful': '😰', 'surprised': '😲'
}

@st.cache_resource
def load_models():
    """Load pre-trained models"""
    if not TF_AVAILABLE:
        st.warning("⚠️ TensorFlow not available. Running in demo mode.")
        return None, None, False
    
    try:
        fer_model = tf.keras.models.load_model('fer_model.h5')
        ser_model = tf.keras.models.load_model('ser_model.h5')
        return fer_model, ser_model, True
    except Exception as e:
        st.warning(f"⚠️ Pre-trained models not found: {str(e)}. Using demo mode with random predictions.")
        return None, None, False

def preprocess_image(image):
    """Preprocess uploaded image for FER model"""
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_array
    
    # Resize to model input size
    img_resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
    
    # Normalize and reshape
    img_normalized = img_resized / 255.0
    img_final = img_normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    
    return img_final

def extract_audio_features(audio_file):
    """Extract MFCC features from audio file"""
    try:
        # Load audio file
        audio, sr = librosa.load(audio_file, res_type='kaiser_fast', sr=SAMPLE_RATE)
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=MFCC_COUNT)
        
        # Pad or truncate to fixed length
        if mfccs.shape[1] > MAX_PAD_LEN:
            mfccs = mfccs[:, :MAX_PAD_LEN]
        else:
            pad_width = MAX_PAD_LEN - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
        # Reshape for model input
        mfccs_final = mfccs.reshape(1, MFCC_COUNT, MAX_PAD_LEN, 1)
        
        return mfccs_final
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None

def predict_emotion(fer_model, ser_model, fer_input, ser_input=None, models_loaded=True):
    """Make predictions using loaded models"""
    if not models_loaded:
        # Demo mode with random predictions
        fer_pred_idx = np.random.randint(0, 7)
        fer_confidence = np.random.uniform(0.6, 0.95)
        fer_emotion = FER_EMOTION_LABELS[fer_pred_idx]
        
        if ser_input is not None:
            ser_pred_idx = np.random.randint(0, 8)
            ser_confidence = np.random.uniform(0.6, 0.95)
            ser_emotion = RAVDESS_EMOTION_LABELS[str(ser_pred_idx + 1).zfill(2)]
            return fer_emotion, fer_confidence, ser_emotion, ser_confidence
        else:
            return fer_emotion, fer_confidence, None, None
    
    # Real predictions
    fer_pred_probs = fer_model.predict(fer_input, verbose=0)[0]
    fer_pred_idx = np.argmax(fer_pred_probs)
    fer_emotion = FER_EMOTION_LABELS[fer_pred_idx]
    fer_confidence = fer_pred_probs[fer_pred_idx]
    
    if ser_input is not None and ser_model is not None:
        ser_pred_probs = ser_model.predict(ser_input, verbose=0)[0]
        ser_pred_idx = np.argmax(ser_pred_probs)
        ser_emotion = RAVDESS_EMOTION_LABELS[str(ser_pred_idx + 1).zfill(2)]
        ser_confidence = ser_pred_probs[ser_pred_idx]
        return fer_emotion, fer_confidence, ser_emotion, ser_confidence
    
    return fer_emotion, fer_confidence, None, None

def get_final_emotion(fer_emotion, fer_conf, ser_emotion, ser_conf):
    """Determine final emotion using weighted approach"""
    if ser_emotion is None:
        return fer_emotion, "Face-only prediction", fer_conf
    
    # Normalize emotion names for comparison
    fer_normalized = fer_emotion.lower()
    ser_normalized = ser_emotion.lower()
    
    # High agreement case
    if fer_conf > 0.6 and ser_conf > 0.6 and fer_normalized == ser_normalized:
        return fer_emotion, "High agreement between face and speech", (fer_conf + ser_conf) / 2
    
    # Weighted decision
    if fer_conf > ser_conf:
        return fer_emotion, f"Weighted towards facial expression ({fer_conf:.1%} confidence)", fer_conf
    else:
        return ser_emotion.capitalize(), f"Weighted towards speech tone ({ser_conf:.1%} confidence)", ser_conf

# Main App
def main():
    # Check if TensorFlow is available
    if not TF_AVAILABLE:
        st.error("🚫 TensorFlow is not properly installed. Please fix the protobuf compatibility issue.")
        st.markdown("""
        ### Quick Fix:
        Run one of these commands in your terminal:
        
        ```bash
        # Option 1 (Recommended)
        pip uninstall protobuf
        pip install protobuf==3.20.3
        
        # Option 2
        pip install --upgrade protobuf==4.23.4
        
        # Option 3
        pip uninstall tensorflow protobuf
        pip install tensorflow==2.15.0 protobuf==4.23.4
        ```
        
        Then restart the Streamlit app.
        """)
        return
    
    # Header
    st.markdown("<h1>🎭 Multimodal Emotion Recognition System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748b; font-size: 18px;'>Analyze emotions through facial expressions and speech patterns</p>", unsafe_allow_html=True)
    
    # Load models
    fer_model, ser_model, models_loaded = load_models()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎯 Mode Selection")
        mode = st.radio(
            "Choose Recognition Mode:",
            ["Face-only Mode", "Multimodal Mode"],
            help="Select whether to analyze face only or both face and speech"
        )
        
        st.markdown("---")
        st.markdown("## 📚 How It Works")
        
        if mode == "Face-only Mode":
            st.markdown("""
            **Face-only Mode** analyzes:
            - Facial expressions
            - Micro-expressions
            - Visual cues
            
            **Supported Emotions:**
            - 😊 Happy
            - 😢 Sad
            - 😠 Angry
            - 😲 Surprise
            - 😨 Fear
            - 🤢 Disgust
            - 😐 Neutral
            """)
        else:
            st.markdown("""
            **Multimodal Mode** combines:
            - Facial expression analysis
            - Speech tone recognition
            - Integrated decision making
            
            **Enhanced Accuracy** through:
            - Cross-validation
            - Confidence weighting
            - Context awareness
            """)
        
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.info("""
        - Use clear, well-lit images
        - Audio files should be 2-5 seconds
        - Supported formats: JPG, PNG, WAV, MP3
        """)
    
    # Main content area
    if mode == "Face-only Mode":
        st.markdown("## 📸 Face-only Emotion Recognition")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Upload Image")
            uploaded_image = st.file_uploader(
                "Choose an image file",
                type=['jpg', 'jpeg', 'png'],
                help="Upload a clear image of a face"
            )
            
            if uploaded_image is not None:
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            if uploaded_image is not None:
                if st.button("🔍 Analyze Emotion", type="primary"):
                    with st.spinner("Analyzing facial expression..."):
                        # Preprocess image
                        processed_img = preprocess_image(image)
                        
                        # Predict
                        fer_emotion, fer_conf, _, _ = predict_emotion(
                            fer_model, ser_model, processed_img, 
                            models_loaded=models_loaded
                        )
                        
                        # Display results
                        st.markdown("### 🎯 Prediction Results")
                        
                        emoji = EMOTION_EMOJIS.get(fer_emotion.lower(), "😊")
                        st.markdown(f"""
                        <div class="result-card">
                            <h1 style="font-size: 4rem; margin: 0;">{emoji}</h1>
                            <h2 style="margin: 1rem 0;">{fer_emotion.upper()}</h2>
                            <p style="font-size: 1.2rem;">Confidence: {fer_conf:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Confidence breakdown
                        st.markdown("### 📊 Confidence Score")
                        st.progress(fer_conf)
                        
                        if fer_conf > 0.8:
                            st.success("✅ High confidence prediction")
                        elif fer_conf > 0.6:
                            st.info("ℹ️ Moderate confidence prediction")
                        else:
                            st.warning("⚠️ Low confidence - results may vary")
    
    else:  # Multimodal Mode
        st.markdown("## 🎭 Multimodal Emotion Recognition")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📸 Upload Image")
            uploaded_image = st.file_uploader(
                "Choose an image file",
                type=['jpg', 'jpeg', 'png'],
                key="multi_image"
            )
            
            if uploaded_image is not None:
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.markdown("### 🎵 Upload Audio")
            uploaded_audio = st.file_uploader(
                "Choose an audio file",
                type=['wav', 'mp3'],
                key="multi_audio"
            )
            
            if uploaded_audio is not None:
                st.audio(uploaded_audio, format='audio/wav')
        
        if uploaded_image is not None and uploaded_audio is not None:
            if st.button("🚀 Analyze Both Modalities", type="primary"):
                with st.spinner("Analyzing facial expression and speech..."):
                    # Preprocess inputs
                    processed_img = preprocess_image(image)
                    processed_audio = extract_audio_features(uploaded_audio)
                    
                    if processed_audio is not None:
                        # Predict
                        fer_emotion, fer_conf, ser_emotion, ser_conf = predict_emotion(
                            fer_model, ser_model, processed_img, processed_audio,
                            models_loaded=models_loaded
                        )
                        
                        # Get final decision
                        final_emotion, reasoning, final_conf = get_final_emotion(
                            fer_emotion, fer_conf, ser_emotion, ser_conf
                        )
                        
                        # Display results
                        st.markdown("---")
                        st.markdown("## 📊 Analysis Results")
                        
                        # Individual predictions
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### 👤 Facial Analysis")
                            emoji = EMOTION_EMOJIS.get(fer_emotion.lower(), "😊")
                            st.markdown(f"""
                            <div class="metric-card">
                                <h2 style="text-align: center; font-size: 3rem;">{emoji}</h2>
                                <h3 style="text-align: center; color: #334155;">{fer_emotion}</h3>
                                <p style="text-align: center; color: #64748b;">Confidence: {fer_conf:.1%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            st.progress(fer_conf)
                        
                        with col2:
                            st.markdown("### 🎤 Speech Analysis")
                            emoji = EMOTION_EMOJIS.get(ser_emotion.lower(), "😊")
                            st.markdown(f"""
                            <div class="metric-card">
                                <h2 style="text-align: center; font-size: 3rem;">{emoji}</h2>
                                <h3 style="text-align: center; color: #334155;">{ser_emotion.capitalize()}</h3>
                                <p style="text-align: center; color: #64748b;">Confidence: {ser_conf:.1%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            st.progress(ser_conf)
                        
                        # Final decision
                        st.markdown("### 🎯 Final Decision")
                        final_emoji = EMOTION_EMOJIS.get(final_emotion.lower(), "😊")
                        st.markdown(f"""
                        <div class="result-card">
                            <h1 style="font-size: 5rem; margin: 0;">{final_emoji}</h1>
                            <h1 style="margin: 1rem 0;">{final_emotion.upper()}</h1>
                            <p style="font-size: 1.3rem; opacity: 0.9;">{reasoning}</p>
                            <p style="font-size: 1.1rem;">Overall Confidence: {final_conf:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Explanation
                        st.markdown("### 💡 How We Decided")
                        st.markdown(f"""
                        <div class="info-box">
                            <p><strong>Decision Logic:</strong></p>
                            <ul>
                                <li>Facial prediction: {fer_emotion} ({fer_conf:.1%})</li>
                                <li>Speech prediction: {ser_emotion.capitalize()} ({ser_conf:.1%})</li>
                                <li>Method: {reasoning}</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
        
        elif uploaded_image is not None or uploaded_audio is not None:
            st.info("📋 Please upload both an image and an audio file to perform multimodal analysis.")

if __name__ == "__main__":
    main()
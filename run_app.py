# run_app.py

import numpy as np
import tensorflow as tf
import cv2
import librosa
import os
from tensorflow.keras.models import load_model

# --- Configuration Constants (Must match the notebook) ---
FER_EMOTION_LABELS = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
RAVDESS_EMOTION_LABELS = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad', 
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}
IMG_SIZE = 48
MFCC_COUNT = 40
MAX_PAD_LEN = 174

def multimodal_predict(fer_model, ser_model, fer_input, ser_input):
    """
    Combines predictions from FER and SER models.
    
    Args:
        fer_model: Trained Keras model for FER.
        ser_model: Trained Keras model for SER.
        fer_input: Preprocessed image array (1, 48, 48, 1).
        ser_input: Preprocessed MFCC array (1, 40, 174, 1).
        
    Returns:
        A dictionary with the final combined emotion prediction.
    """
    # 1. Individual Predictions
    fer_pred_probs = fer_model.predict(fer_input, verbose=0)[0]
    ser_pred_probs = ser_model.predict(ser_input, verbose=0)[0]
    
    fer_pred_index = np.argmax(fer_pred_probs)
    ser_pred_index = np.argmax(ser_pred_probs)
    
    fer_emotion = FER_EMOTION_LABELS.get(fer_pred_index, "Unknown")
    # Adjust index to RAVDESS keys (1-indexed, 2-digit string)
    ser_emotion = RAVDESS_EMOTION_LABELS.get(str(ser_pred_index + 1).zfill(2), "Unknown") 
    
    return {
        "facial_emotion": fer_emotion,
        "facial_confidence": fer_pred_probs[fer_pred_index],
        "speech_emotion": ser_emotion,
        "speech_confidence": ser_pred_probs[ser_pred_index],
    }

def run_cli_demo():
    """
    The main function for the Command-Line Interface (CLI) application.
    It simulates a real-time prediction using pre-trained models.
    """
    print("\n==================================================")
    print("  Multimodal Emotion Recognition CLI Application  ")
    print("==================================================")
    
    # --- Load Models ---
    try:
        # NOTE: In a real scenario, the user must ensure 'fer_model.h5' and 'ser_model.h5' exist
        # by running the training part of the notebook first.
        fer_m = load_model('fer_model.h5')
        ser_m = load_model('ser_model.h5')
        print("Models loaded successfully.")
    except Exception as e:
        print(f"ERROR: Could not load models. Please ensure 'fer_model.h5' and 'ser_model.h5' are trained and saved.")
        print(f"Error details: {e}")
        print("Exiting application.")
        return

    # --- Simulation of real-world input ---
    print("\nSimulating input from a live video/audio stream...")
    
    # 1. Simulate Image Capture and Preprocessing
    # In a real application, this would capture a frame from a webcam and detect a face.
    # We use a synthetic input for demonstration.
    print("1. Processing Facial Input (Simulated Frame)...")
    # Create a synthetic 48x48 grayscale image (normalized)
    simulated_fer_input = np.random.rand(1, IMG_SIZE, IMG_SIZE, 1).astype('float32')
    
    # 2. Simulate Audio Capture and Feature Extraction
    # We use a synthetic MFCC feature array for demonstration.
    print("2. Processing Speech Input (Simulated Audio Clip)...")
    # Create a synthetic MFCC array (40 features x 174 time steps)
    simulated_ser_input = np.random.rand(1, MFCC_COUNT, MAX_PAD_LEN, 1).astype('float32')
    
    # 3. Run Multimodal Prediction
    print("3. Running Multimodal Prediction...")
    result = multimodal_predict(fer_m, ser_m, simulated_fer_input, simulated_ser_input)
    
    # 4. Display Results
    print("\n--- Final Multimodal Result ---")
    print(f"Facial Emotion: {result['facial_emotion']} (Confidence: {result['facial_confidence']:.2f})")
    print(f"Speech Emotion: {result['speech_emotion']} (Confidence: {result['speech_confidence']:.2f})")
    
    # Simple Fusion Logic for Final Output
    # This logic prioritizes agreement and then the higher confidence score.
    
    # Check for high agreement on a common emotion (Happy, Sad, Angry, Fearful, Neutral)
    common_emotions = ['Happy', 'Sad', 'Angry', 'Fearful', 'Neutral']
    fer_emo = result['facial_emotion']
    ser_emo = result['speech_emotion']
    
    if fer_emo in common_emotions and ser_emo in common_emotions and fer_emo == ser_emo:
        final_emotion = fer_emo
        print(f"\nFINAL DECISION (High Agreement on {final_emotion}): {final_emotion}")
    elif result['facial_confidence'] > result['speech_confidence']:
        final_emotion = fer_emo
        print(f"\nFINAL DECISION (Weighted towards Face, Confidence: {result['facial_confidence']:.2f}): {final_emotion}")
    else:
        final_emotion = ser_emo
        print(f"\nFINAL DECISION (Weighted towards Speech, Confidence: {result['speech_confidence']:.2f}): {final_emotion}")
        
    print("==================================================")

if __name__ == '__main__':
    # Suppress TensorFlow warnings for cleaner output
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    run_cli_demo()

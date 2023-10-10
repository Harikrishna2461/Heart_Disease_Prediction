import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from scipy.signal import chirp
from PIL import Image
import cv2
import base64
import os
from scipy.io import wavfile
import matplotlib.pyplot as plt

bgm_path = os.path.join(os.path.dirname(__file__), "White_BGM.png")

# Define set_bg_hack function without st.markdown
def set_bg_hack(main_bg):
    main_bg_ext = "png"
    return main_bg_ext

# Call set_bg_hack and use st.markdown outside the function
main_bg_ext = set_bg_hack(bgm_path)
st.markdown(
    f"""
    <style>
    .stApp {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(bgm_path, "rb").read()).decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Load the TensorFlow Lite model
model_path = os.path.join(os.path.dirname(__file__),"autokeras_synch_sft_model.tflite")
interpreter = tf.lite.Interpreter(model_path)
interpreter.allocate_tensors()

# Functions for preprocessing sound wave into an image
# Synchrosqueezing
def synchrosqueeze(matrix, t):
    inst_freq = np.abs(np.gradient(np.angle(matrix), axis=0))
    synchrosqueezed = np.zeros_like(matrix, dtype=complex)
    for i in range(matrix.shape[0]):
        synchrosqueezed[i, :] = matrix[i, :] * np.exp(2j * np.pi * t * inst_freq[i, :])
    return synchrosqueezed
    
def SFT(path, window_sizes):
    sample_rate, audio_data = wavfile.read(path)

    results = []
    for fft_size in window_sizes:
        overlap_fac = 0.5
        hop_size = np.int32(np.floor(fft_size * (1 - overlap_fac)))
        pad_end_size = fft_size
        total_segments = np.int32(np.ceil(len(audio_data) / np.float32(hop_size)))
        t_max = len(audio_data) / np.float32(sample_rate)

        window = np.hanning(fft_size)
        inner_pad = np.zeros(fft_size)

        proc = np.concatenate((audio_data, np.zeros(pad_end_size)))
        result = np.empty((total_segments, fft_size), dtype=np.float32)

        for i in range(total_segments):
            current_hop = hop_size * i
            segment = proc[current_hop:current_hop + fft_size]
            windowed = segment * window
            padded = np.append(windowed, inner_pad)
            spectrum = np.fft.fft(padded) / fft_size
            autopower = np.abs(spectrum * np.conj(spectrum))
            result[i, :] = autopower[:fft_size]

        result = 20 * np.log10(result)
        result = np.clip(result, -40, 200)

        synchrosqueezed_result = synchrosqueeze(result.T, np.arange(result.shape[0]))

        resized_spectrogram = cv2.resize(np.abs(synchrosqueezed_result), (128, 128), interpolation=cv2.INTER_LINEAR)
        resized_spectrogram = (255 * (resized_spectrogram - resized_spectrogram.min()) / (resized_spectrogram.max() - resized_spectrogram.min())).astype(np.uint8)
        colormap = plt.get_cmap('viridis')
        spectrogram_rgb = colormap(resized_spectrogram)[:, :, :3]

        results.append(spectrogram_rgb)

    return results


# Streamlit UI
st.title(":green[Heart Disease Prediction from Heart Beat Sound Wave]")

# Upload sound wave file
uploaded_file = st.file_uploader(":green[Upload a sound wave file (.wav)]", type=["wav"])

# Create a button to predict the disease
predict_button = st.button("Predict the Disease")

if uploaded_file and predict_button:
    try:
        # Preprocess the sound wave into an image
        window_sizes = [512]
        input_image = SFT(uploaded_file, window_sizes)

        if predict_button:  # Check if the button is pressed
            # Make predictions using the TensorFlow Lite model
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            interpreter.set_tensor(input_details[0]['index'], np.expand_dims(input_image, axis=0))
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            predicted_class = np.argmax(output)

            # Define your class labels here
            class_labels = ["Aortic Stenosis", "Mitral Regurgitation", "Mitral Stenosis", "Mitral Valve Prolapse", "No Disease"]

            # Display the predicted class
            if predicted_class != 4:
                st.success(f"**The Predicted Disease is : {class_labels[predicted_class]}**")
            else:
                st.success(f"**The Patient has {class_labels[predicted_class]}**")

    except Exception as e:
        st.error(f"Error: {str(e)}")
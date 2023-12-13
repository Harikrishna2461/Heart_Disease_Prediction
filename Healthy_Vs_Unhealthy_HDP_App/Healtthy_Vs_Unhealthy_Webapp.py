import streamlit as st
import numpy as np
import tensorflow as tf
from scipy.signal import chirp
from PIL import Image
import cv2
import base64
import os
from scipy.io import wavfile
import matplotlib.pyplot as plt

#bgm_path = os.path.join(os.path.dirname(__file__), "White_BGM.png")

# Define set_bg_hack function without st.markdown
#def set_bg_hack(main_bg):
  #  main_bg_ext = "png"
  # return main_bg_ext

# Call set_bg_hack and use st.markdown outside the function
#main_bg_ext = set_bg_hack(bgm_path)
#st.markdown(
    #f"""
    #<style>
    #.stApp {{
    #    background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(bgm_path, "rb").read()).decode()});
     #   background-size: cover
   # }}
    #</style>
    #""",
    #unsafe_allow_html=True
#)

# Load the TensorFlow Lite model
model_path = os.path.join(os.path.dirname(__file__), "Finalised_MobileNetV2_SFT_Model_Dataset-2.tflite")
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

def SFT(path, window_size):
    sample_rate, audio_data = wavfile.read(path)

    overlap_fac = 0.5
    hop_size = np.int32(np.floor(window_size * (1 - overlap_fac)))
    pad_end_size = window_size
    total_segments = np.int32(np.ceil(len(audio_data) / np.float32(hop_size)))
    t_max = len(audio_data) / np.float32(sample_rate)

    window = np.hanning(window_size)
    inner_pad = np.zeros(window_size)

    proc = np.concatenate((audio_data, np.zeros(pad_end_size)))
    result = np.empty((total_segments, window_size), dtype=np.float32)

    for i in range(total_segments):
        current_hop = hop_size * i
        segment = proc[current_hop:current_hop + window_size]
        windowed = segment * window
        padded = np.append(windowed, inner_pad)
        spectrum = np.fft.fft(padded) / window_size
        autopower = np.abs(spectrum * np.conj(spectrum))
        result[i, :] = autopower[:window_size]
        
    result = 20 * np.log10(result)  
    result = np.clip(result, -40, 200)

    synchrosqueezed_result = synchrosqueeze(result.T, np.arange(result.shape[0]))

    resized_spectrogram = cv2.resize(np.abs(synchrosqueezed_result), (128, 128), interpolation=cv2.INTER_LINEAR)
    resized_spectrogram = (255 * (resized_spectrogram - resized_spectrogram.min()) / (resized_spectrogram.max() - resized_spectrogram.min())).astype(np.uint8)
    colormap = plt.get_cmap('viridis')
    spectrogram_rgb = colormap(resized_spectrogram)[:, :, :3]

    return spectrogram_rgb

# Streamlit UI
st.title(":blue[Patient Heart Health Condition Prediction Using Heart Beat Sound Data]")

# Upload sound wave file
uploaded_file = st.file_uploader(":green[Upload a sound wave file (.wav)]", type=["wav"])

# Create a button to predict the disease
predict_button = st.button("Predict the Disease")

if uploaded_file and predict_button:
    try:
        # Preprocess the sound wave into an image
        window_size = 512
        input_image = SFT(uploaded_file, window_size)

        if predict_button:  # Check if the button is pressed
            # Make predictions using the TensorFlow Lite model
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            # Convert the input image data to FLOAT32
            input_image = input_image.astype('float32')
            # Reshape the input image to match the expected shape (batch_size=1, height, width, channels=3)
            input_image = np.expand_dims(input_image, axis=0)

            interpreter.set_tensor(input_details[0]['index'], input_image)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            predicted_class = np.argmax(output)

            # Define your class labels here
            class_labels = ["Healthy","Unhealthy"]
            st.markdown(
                        f'<p style="color:black; background-color:green; font-size:20px;">'
                        f'<strong>The Patient is {class_labels[predicted_class]}</strong>'
                        f'</p>',
                        unsafe_allow_html=True
                        )

    except Exception as e:
        st.error(f"Error: {str(e)}")

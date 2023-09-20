import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from scipy.signal import chirp
from PIL import Image
import base64
import os

bgm_path = os.path.join(os.path.dirname(__file__), "bgm.png")
# Define set_bg_hack function without st.markdown
def set_bg_hack(main_bg):
    main_bg_ext = "png"
    return main_bg_ext

#Call set_bg_hack and use st.markdown outside the function
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
#set_bg_hack(bgm_path)

# Load the TensorFlow Lite model
model_path = os.path.join(os.path.dirname(__file__),"model_int8_latest.tflite")
interpreter = tf.lite.Interpreter(model_path)
interpreter.allocate_tensors()

# Function for preprocessing sound wave into an image
def sound_data_to_image_loading_and_preprocessing_chirplet(file_path):
    # Load audio file and perform Mel spectrogram conversion
    sample_s, sr = librosa.load(file_path, sr=32000, mono=True)
    melspec = librosa.feature.melspectrogram(y=sample_s, sr=sr)
    melspec = librosa.power_to_db(melspec).astype(np.float32)
    
    # Apply chirplet transformation
    w = chirp(melspec, f0=6, f1=1, t1=10, method='linear')
    
    # Normalize values to range [0, 255]
    w_min, w_max = np.min(w), np.max(w)
    w = 255.0 * (w - w_min) / (w_max - w_min)
    
    # Convert the NumPy array to a Pillow image
    img1 = Image.fromarray(w.astype('uint8'), mode='L')
    
    # Resize the image to 128x128 pixels using Lanczos resampling
    img1 = img1.resize((128, 128), Image.LANCZOS)
    
    # Convert the Pillow image back to a NumPy array
    resized_img = np.array(img1)
    
    # Ensure the array has a shape of (128, 128, 1)
    resized_img = resized_img.reshape((128, 128, 1))
    
    return resized_img

# Streamlit UI
st.title(":red[Sound Wave to Disease Prediction]")
# Change the title color to red
#st.markdown("<span style='color:red'>Custom Color Title</span>", unsafe_allow_html=True)


# Upload sound wave file
uploaded_file = st.file_uploader(":green[Upload a sound wave file (.wav)]", type=["wav"])

# Create a button to predict the disease
predict_button = st.button("Predict the Disease")

if uploaded_file and predict_button:
    try:
        # Preprocess the sound wave into an image
        input_image = sound_data_to_image_loading_and_preprocessing_chirplet(uploaded_file)

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

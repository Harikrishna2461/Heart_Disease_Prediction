import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from scipy.signal import chirp
from PIL import Image
import cv2
import base64

@st.cache_data
def set_bg_hack(main_bg):
    '''
    A function to unpack an image from root folder and set as bg.
 
    Returns
    -------
    The background.
    '''
    # set bg name
    main_bg_ext = "png"
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

set_bg_hack('bgm.png')

# Load the TensorFlow Lite model
model_path = 'C:/Users/Hari Krishna D/OneDrive/Desktop/Projects/Heart Disease Prediction Using Self Supervised Neural Networks Project/TFLite Saved Models/model_int8_latest.tflite'
interpreter = tf.lite.Interpreter(model_path)
interpreter.allocate_tensors()

# Function for preprocessing sound wave into an image
def sound_data_to_image_loading_and_preprocessing_chirplet(file_path):
    sample_s, sr = librosa.load(file_path, sr=32000, mono=True)
    melspec = librosa.feature.melspectrogram(y=sample_s, sr=sr)
    melspec = librosa.power_to_db(melspec).astype(np.float32)
    w = chirp(melspec, f0=6, f1=1, t1=10, method='linear')
    Height = w.shape[0]
    Width = w.shape[1]
    img1 = Image.fromarray(w, "I")
    img1 = np.array(img1)
    arr1 = img1.reshape((Height, Width, 1))
    arr2 = np.array(arr1, dtype='uint8')
    resized_img = cv2.resize(arr2, (128, 128))
    resized_img = resized_img.reshape((128, 128, 1))
    c_rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB)
    c_rgb_img = np.array(c_rgb_img)
    return c_rgb_img

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

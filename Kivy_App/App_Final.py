# Import necessary libraries and modules
from kivy.logger import Logger
Logger.setLevel('INFO')
Logger.info("MyApp: Starting the application")

import os
import tensorflow as tf
import numpy as np
import cv2
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.io import wavfile
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.widget import Widget
from kivy.app import App


class MyApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical',padding=10, spacing=10)
        # Create a horizontal layout for buttons
        self.button_layout = BoxLayout(orientation='horizontal', spacing=300, size_hint=(None,None), size=(500,500))
        self.button_layout.center_x = self.layout.center_x
        self.button_layout.size_hint_x = 1 # Make it cover the entire width
        #self.file_chooser = FileChooserListView()
        #self.file_chooser = BoxLayout(orientation='vertical')  # Initialize as an empty widget
        #self.file_chooser = FileChooserListView(path=os.getcwd(), dirselect=False)
        #self.file_chooser.opacity = 0  # Make it fully transparent initially
        #Creating the Text
        self.output_text = TextInput(readonly=True,size_hint_y=None, height=300)
        #Creating the Butoons
        self.upload_button = Button(text="Upload File", on_release=self.create_file_chooser,size=(500, 500))
        self.predict_button = Button(text="Predict", on_release=self.predict,size=(500, 500))
  
        # Initialize the file chooser but keep it hidden
        #self.file_chooser = FileChooserListView(path='', dirselect=False)
        #self.file_chooser.bind(on_submit=self.file_selected)
        #self.file_chooser.opacity = 0
        #self.layout.add_widget(self.file_chooser)
        
        # Add spacer widgets to center buttons vertically
        spacer1 = Widget()
        spacer2 = Widget()

        self.button_layout.add_widget(spacer1)
        self.button_layout.add_widget(self.upload_button)
        #self.button_layout.add_widget(spacer2)
        self.button_layout.add_widget(self.predict_button)
        self.button_layout.add_widget(spacer2)

        
        #self.layout.add_widget(self.upload_button)
        #self.layout.add_widget(self.predict_button)
        self.layout.add_widget(Widget())  # Spacer widget to push buttons to the center
        self.layout.add_widget(self.button_layout)
        self.layout.add_widget(Widget())  # Spacer widget to push buttons to the center
        self.layout.add_widget(self.output_text)

        #self.file_chooser = None

        return self.layout
    
    def create_file_chooser(self, instance):
        if hasattr(self, 'file_chooser'):
            self.layout.remove_widget(self.file_chooser)
        self.file_chooser = FileChooserListView(path='', dirselect=False)
        self.file_chooser.bind(on_submit=self.file_selected)
        self.layout.add_widget(self.file_chooser)

    #def toggle_file_chooser(self, instance):
      #  if self.file_chooser.opacity == 0:
       #     self.file_chooser.opacity = 1  # Make it fully opaque (visible)
        #    self.file_chooser.bind(on_submit=self.file_selected)  # Bind the file_selected method
        #else:
         #   self.file_chooser.opacity = 0  # Make it fully transparent (hidden)

    #def file_selected(self, instance , selection, touch):
     #   if self.file_chooser.selection:
      #      selected_file = self.file_chooser.selection[0]
            # Preprocess the selected .wav file and store it as an image
       #     image = self.SFT(selected_file)
        #    self.image_data = image
         #   self.file_chooser.unbind(on_submit=self.file_selected)
    def file_selected(self, instance, selection, touch):
        if selection:
            selected_file = selection[0]
            image = self.SFT(selected_file)
            self.image_data = image
            self.layout.remove_widget(self.file_chooser)

    def predict(self, instance):
        if hasattr(self, 'image_data'):
        # Load your TensorFlow Lite model here
            model = tf.lite.Interpreter(model_path='Finalised_MobileNetV2_SFT_Model.tflite')
            model.allocate_tensors()

            # Get input details
            input_details = model.get_input_details()
            input_shape = input_details[0]['shape']

            # Prepare the input data
            input_data = np.expand_dims(self.image_data, axis=0).astype(np.float32)

            # Check if the input data shape matches the expected shape
            if input_data.shape == tuple(input_shape):
                input_tensor_index = input_details[0]['index']
                model.set_tensor(input_tensor_index, input_data)

                # Run the inference
                model.invoke()

                # Get the output and display it
                output_tensor_index = model.get_output_details()[0]['index']
                output_data = model.get_tensor(output_tensor_index)
                predicted_class = np.argmax(output_data)
                # Define your class labels here
                class_labels = ["Aortic Stenosis", "Mitral Regurgitation", "Mitral Stenosis", "Mitral Valve Prolapse", "No Disease"]
                if predicted_class != 4:
                    prediction_result = f'The Predicted Disease is : {class_labels[predicted_class]}' 
                else:
                    prediction_result = f'The Patient has : {class_labels[predicted_class]}' 

                self.output_text.text = prediction_result
            else:
                self.output_text.text = 'Input data shape does not match the model requirements.'
        else:
            self.output_text.text = 'No file selected.'

        

    # Synchrosqueezing
    def synchrosqueeze(self, matrix, t):
        inst_freq = np.abs(np.gradient(np.angle(matrix), axis=0))
        synchrosqueezed = np.zeros_like(matrix, dtype=complex)
        for i in range(matrix.shape[0]):
            synchrosqueezed[i, :] = matrix[i, :] * np.exp(2j * np.pi * t * inst_freq[i, :])
        return synchrosqueezed

    def SFT(self, path):
        # Load the audio file
        sample_rate, audio_data = wavfile.read(path)

        # Define SFT parameters
        fft_size = 512
        overlap_fac = 0.5

        hop_size = np.int32(np.floor(fft_size * (1 - overlap_fac)))
        pad_end_size = fft_size
        total_segments = np.int32(np.ceil(len(audio_data) / np.float32(hop_size)))
        t_max = len(audio_data) / np.float32(sample_rate)

        window = np.hanning(fft_size)
        inner_pad = np.zeros(fft_size)

        proc = np.concatenate((audio_data, np.zeros(pad_end_size)))
        result = np.empty((total_segments, fft_size), dtype=np.float32)

        # Compute SFT
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

        synchrosqueezed_result = self.synchrosqueeze(result.T, np.arange(result.shape[0]))

        resized_spectrogram = cv2.resize(np.abs(synchrosqueezed_result), (128, 128), interpolation=cv2.INTER_LINEAR)
        # Normalize the spectrogram to values between 0 and 255
        resized_spectrogram = (255 * (resized_spectrogram - resized_spectrogram.min()) / (resized_spectrogram.max() - resized_spectrogram.min())).astype(np.uint8)
        # Create an RGB image from the spectrogram
        colormap = plt.get_cmap('viridis')
        spectrogram_rgb = colormap(resized_spectrogram)[:, :, :3]

        return spectrogram_rgb

if __name__ == '__main__':
    MyApp().run()

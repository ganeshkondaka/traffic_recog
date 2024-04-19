import os
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
from PIL import Image
import pickle

from IPython.display import HTML
import streamlit as st
from gtts import gTTS
from pydub import AudioSegment
from io import BytesIO

# Dictionary mapping class indices to class names
classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)',
            2:'Speed limit (50km/h)',
            3:'Speed limit (60km/h)',
            4:'Speed limit (70km/h)',
            5:'Speed limit (80km/h)',
            6:'End of speed limit (80km/h)',
            7:'Speed limit (100km/h)',
            8:'Speed limit (120km/h)',
            9:'No passing',
            10:'No passing veh over 3.5 tons',
            11:'Right-of-way at intersection',
            12:'Priority road',
            13:'Yield',
            14:'Stop',
            15:'No vehicles',
            16:'Vehicle > 3.5 tons prohibited',
            17:'No entry',
            18:'General caution',
            19:'Dangerous curve left',
            20:'Dangerous curve right',
            21:'Double curve',
            22:'Bumpy road',
            23:'Slippery road',
            24:'Road narrows on the right',
            25:'Road work',
            26:'Traffic signals',
            27:'Pedestrians',
            28:'Children crossing',
            29:'Bicycles crossing',
            30:'Beware of ice/snow',
            31:'Wild animals crossing',
            32:'End speed + passing limits',
            33:'Turn right ahead',
            34:'Turn left ahead',
            35:'Ahead only',
            36:'Go straight or right',
            37:'Go straight or left',
            38:'Keep right',
            39:'Keep left',
            40:'Roundabout mandatory',
            41:'End of no passing',
            42:'End no passing veh > 3.5 tons' }


#D:\mine_\code files @@\pythonf\traffic_proj

from keras.models import load_model

# Load the model
loaded_model = load_model('D:/mine_/code files @@/pythonf/traffic_proj/modelh.h5')


# Load the pre-trained model
# loaded_model = pickle.load(open('D:/mine_/code files @@/pythonf/traffic_proj/sign_model.sav', 'rb'))

# import joblib

# # Load the model
# loaded_model = joblib.load('D:/mine_/code files @@/pythonf/traffic_proj/modelh.h5')


# Function to make predictions
def predict_traffic_sign(image):
    # Preprocess the image
    image = image.resize((30, 30))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    # Predict using the loaded model
    prediction = np.argmax(loaded_model.predict(image))
    return classes[prediction]

# Function to convert text to audio and embed it in HTML
def text_to_audio(text, language='en'):
    # Generate audio from text
    tts = gTTS(text=text, lang=language, slow=False)
    # Save audio as a temporary file
    audio_path = "prediction_audio.mp3"
    tts.save(audio_path)
    # Create HTML audio player with autoplay
    audio_html = f'<audio autoplay="autoplay" controls="controls"><source src="{audio_path}" type="audio/mp3"></audio>'
    return audio_html


def main():
    st.title("Traffic Sign Recognition App")

    st.write("Upload an image of a traffic sign to predict its class.")

    # File uploader
    uploaded_file = st.file_uploader("upload your image here......", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', width=400)

        # Make prediction
        prediction = predict_traffic_sign(image)
        st.write("Prediction:", prediction)

        # audio_path = "D:/mine_/code files @@/pythonf/traffic_proj/outputaudio.mp3"

        # # Display the audio player
        # st.audio(audio_path, format='audio/mp3')

        # Example usage
        # Replace this with your prediction value
        st.write("Prediction:", prediction)

        # Convert prediction text to audio and display it
        audio_html = text_to_audio(prediction)
        st.write(HTML(audio_html))



if __name__ == '__main__':
    main()

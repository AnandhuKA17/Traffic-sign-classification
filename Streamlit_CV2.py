import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
import cv2

# Load the trained model
model = load_model("model.h5")


def preprocess_image(image):

    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # Resize the image to match the model's expected sizing
    image = cv2.resize(image, (30, 30))
    image = np.array(image)
    # Normalize the pixel values to be between 0 and 1
    image = image / 255.0
    # Reshape the image to (1, 30, 30, 3) as the model expects a batch of images
    image = np.reshape(image, (1, 30, 30, 3))
    return image

def main():
    st.title("Traffic Sign Classification App")

    uploaded_file = st.file_uploader("Choose a traffic sign image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        st.write("")

        # Convert the image to a NumPy array
        
        image_array = np.array(image)
        # Preprocess the image
        processed_image = preprocess_image(image_array)
        st.image(processed_image[0], caption="Processed Image", use_column_width=True)

        def predictions(processed_image):
            classes = { 1:'Speed limit (20km/h)',
            2:'Speed limit (30km/h)',      
            3:'Speed limit (50km/h)',       
            4:'Speed limit (60km/h)',      
            5:'Speed limit (70km/h)',    
            6:'Speed limit (80km/h)',      
            7:'End of speed limit (80km/h)',     
            8:'Speed limit (100km/h)',    
            9:'Speed limit (120km/h)',     
           10:'No passing',   
           11:'No passing veh over 3.5 tons',     
           12:'Right-of-way at intersection',     
           13:'Priority road',    
           14:'Yield',     
           15:'Stop',       
           16:'No vehicles',       
           17:'Veh > 3.5 tons prohibited',       
           18:'No entry',       
           19:'General caution',     
           20:'Dangerous curve left',      
           21:'Dangerous curve right',   
           22:'Double curve',      
           23:'Bumpy road',     
           24:'Slippery road',       
           25:'Road narrows on the right',  
           26:'Road work',    
           27:'Traffic signals',      
           28:'Pedestrians',     
           29:'Children crossing',     
           30:'Bicycles crossing',       
           31:'Beware of ice/snow',
           32:'Wild animals crossing',      
           33:'End speed + passing limits',      
           34:'Turn right ahead',     
           35:'Turn left ahead',       
           36:'Ahead only',      
           37:'Go straight or right',      
           38:'Go straight or left',      
           39:'Keep right',     
           40:'Keep left',      
           41:'Roundabout mandatory',     
           42:'End of no passing',      
           43:'End no passing veh > 3.5 tons' }
            # Make predictions
            predictions = model.predict(processed_image)
            predicted_class = np.argmax(predictions)
            st.write(f"Predicted Class: {classes[predicted_class+1]}")

    if st.button('Classify'):
        predictions(processed_image)

# Run the app
if __name__ == "__main__":
    main()
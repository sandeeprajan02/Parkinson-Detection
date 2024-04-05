import streamlit as st
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

# Load your trained model
model = load_model('C:/Users/nvraj/Downloads/S8 Project/best_weights.hdf5')  # Update with your model path

# Define class names
class_names = ['Mild Dementia', 'Moderate Dementia', 'Non Dementia', 'Very Mild Dementia']

# Streamlit UI
st.title("Alzheimer's Disease Classification")
st.write("Upload an MRI brain image and we'll predict the disease stage.")

# Image upload
uploaded_image = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    
    # Perform prediction when the user clicks the "Predict" button
    if st.button("Predict"):
        # Load and preprocess the image
        img = load_img(uploaded_image, target_size=(224, 224))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Make predictions using your model
        predictions = model.predict(img)
        
        # Get the class with the highest probability
        predicted_class = np.argmax(predictions)
        probability = round(predictions[0][predicted_class] * 100, 2)
        
        # Display the result
        st.subheader("Prediction Result:")
        st.write(f"Class: {class_names[predicted_class]}")
        st.write(f"Probability: {probability}%")

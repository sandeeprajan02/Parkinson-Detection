from tensorflow import keras
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt

# Load your model
model = keras.models.load_model('C:/Users/nvraj/Downloads/S8 Project/best_weights.hdf5')  # Replace with the path to your saved model

# Load and preprocess the image
img_path = 'C:/Users/nvraj/Downloads/S8 Project/Dementia.jpeg'
img = load_img(img_path, target_size=(224, 224, 3))  # Ensure target_size has 3 channels
img = img_to_array(img) / 255.0
img = np.expand_dims(img, axis=0)

# Make predictions on the image
predictions = model.predict(img)
class_index = np.argmax(predictions)

# Get class probabilities
class_probabilities = predictions[0]

# Get class labels and indices from your dataset
dic = test_dataset.class_indices
idc = {k: v for v, k in dic.items()}

# Get the class label and probability
class_label = idc[class_index]
probability = round(class_probabilities[class_index] * 100, 2)

print(probability, '%c chances are there that the image is', class_label)

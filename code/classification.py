# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Developer details: 
        # Name: Akshat Rastogi, Shubh Gupta and Rupal Mishra
        # Role: Developers
        # Code ownership rights: PreProd Corp
    # Version:
        # Version: V 1.1 (19 September 2024)
            # Developers: Akshat Rastogi, Shubh Gupta and Rupal Mishra
            # Unit test: Pass
            # Integration test: Pass
     
    # Description: This Streamlit app allows users to input features and make predictions using Unsupervised Learning.
        # SQLite: Yes
        # MQs: No
        # Cloud: No
        # Data versioning: No
        # Data masking: No

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Dependency: 
    # Environment:     
        # Python 3.11.5
        # Streamlit 1.36.0

import os
import torch
from train import BirdClassificationCNN
from ingest_transform import preprocess
import streamlit as st
from PIL import Image
import os
import numpy as np
import tensorflow as tf  # For Keras model
from keras.preprocessing.image import img_to_array, load_img

def predict_image_class(model, img_tensor, class_labels):
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_labels[predicted.item()]
        return predicted_class

def classify(image):
    image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        model = BirdClassificationCNN(num_classes=25)
        model.load_state_dict(torch.load('code/saved_model/bird_classification_cnn.pth'))
        model.eval()
        output = model(image_tensor)
        _, predicted_idx = torch.max(output, 1)
    
    data_dir = "data\Master"
    # Map predicted index to class name (you need to define this mapping)
    class_names = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]
    print(class_names)
    
    predicted_class = class_names[predicted_idx.item()]
    return predicted_class


# def classify(image_path):
#     # Load the model from the .h5 file
#     model = tf.keras.models.load_model('code/saved_model/bird_classification_cnn.h5')
    
#     # Load and preprocess the image
#     image = load_img(image_path, target_size=(150, 150))  # Resize to match the model's expected input
#     image_tensor = img_to_array(image)  # Convert to numpy array
#     image_tensor = np.expand_dims(image_tensor, axis=0)  # Add batch dimension

#     # Normalize the image if required (adjust based on your training preprocessing)
#     image_tensor /= 255.0  # Assuming the model was trained on images normalized to [0, 1]
    
#     # Make predictions
#     predictions = model.predict(image_tensor)
#     predicted_idx = np.argmax(predictions, axis=1)

#     # Map predicted index to class name
#     data_dir = r"data\Master"  # Update path to your class folder
#     class_names = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]

#     predicted_class = class_names[predicted_idx[0]]  # Get the predicted class name
#     return predicted_class
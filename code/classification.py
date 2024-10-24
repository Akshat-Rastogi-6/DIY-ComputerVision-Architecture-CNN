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

def predict_image_class(model, img_tensor, class_labels):
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_labels[predicted.item()]
        return predicted_class

# def classify(image):
#     image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
#     with torch.no_grad():
#         model = BirdClassificationCNN(num_classes=25)
#         model.load_state_dict(torch.load('code/saved_model/bird_classification_cnn.pth'))
#         model.eval()
#         output = model(image_tensor)
#         _, predicted_idx = torch.max(output, 1)
    
#     data_dir = "extracted_images"
#     # Map predicted index to class name (you need to define this mapping)
#     class_names = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]
#     print(class_names)
    
#     predicted_class = class_names[predicted_idx.item()]
#     return predicted_class

def classify(image):
    # Initialize the model first
    model = BirdClassificationCNN(num_classes=25)  
    # Load the model's state dictionary
    model.load_state_dict(torch.load('code/saved_model/bird_classification_cnn.pth'))
    model.eval()  # Set the model to evaluation mode
    
    # Preprocess the image
    image_tensor = preprocess(image)  # No need for unsqueeze here, preprocess should return the right shape
    
    # Check shape for debugging
    print(f'Image tensor shape: {image_tensor.shape}')  # This should be [3, 150, 150]
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)  # Now shape should be [1, 3, 150, 150]

    with torch.no_grad():  # Disable gradient tracking for inference
        output = model(image_tensor)  # Forward pass
        _, predicted_idx = torch.max(output, 1)  # Get the index of the max log-probability
    
    data_dir = "extracted_images\Master"
    # Map predicted index to class name (you need to define this mapping)
    class_names = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]

    # Get the predicted class name
    predicted_class = class_names[predicted_idx.item()]  # Assuming class_names is defined
    return predicted_class
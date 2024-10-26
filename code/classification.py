# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Developer details: 
        # Name: Akshat Rastogi, Shubh Gupta
        # Role: Developers
        # Code ownership rights: PreProd Corp
    # Version:
        # Version: V 1.1 (25 September 2024)
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
from torchvision import models, transforms
from train import BirdClassificationCNN  # Import custom CNN model
from ingest_transform import preprocess  # Import preprocessing function

# Function to load the selected model based on user choice
def load_model(model_choice):
    # Check if the user wants to load the self-trained CNN model
    if model_choice == "Self-trained CNN":
        # Instantiate the self-trained CNN model with 25 output classes
        model = BirdClassificationCNN(num_classes=25)
        
        # Load the model weights from a saved file
        model.load_state_dict(torch.load('code/saved_model/bird_classification_cnn.pth'))
    
    # If the user selects the pretrained ResNet model
    elif model_choice == "Pretrained ResNet":
        # Load a pretrained ResNet-18 model from torchvision's models
        model = models.resnet18(pretrained=True)  # Using ResNet18 as an example
        
        # Adjust the final layer of the ResNet model to match the required number of classes (25 classes)
        model.fc = torch.nn.Linear(model.fc.in_features, 25)
        
        # Load custom pretrained weights for the ResNet model
        pretrained_dict = torch.load('code/saved_model/pretrained_resnet.pth')
        
        # Get the current model's state dictionary
        model_dict = model.state_dict()
        
        # Update only the weights that match in both the custom pretrained model and current model
        for key in pretrained_dict.keys():
            if key in model_dict:
                # Ensure that only layers with matching dimensions are updated
                if pretrained_dict[key].size() == model_dict[key].size():
                    model_dict[key] = pretrained_dict[key]
        
        # Load the updated weights into the model
        model.load_state_dict(model_dict)

    # Set the model to evaluation mode, which is necessary for inference to disable dropout and batch normalization updates
    model.eval()
    return model


# Function to predict the class of an image given a trained model and preprocessed tensor
def predict_image_class(model, img_tensor, class_labels):
    with torch.no_grad():  # Disable gradient computation for efficiency
        outputs = model(img_tensor)  # Forward pass: compute the model's output
        _, predicted = torch.max(outputs, 1)  # Get the index of the class with the highest score
        
        # Retrieve the predicted class label from the list of class labels
        predicted_class = class_labels[predicted.item()]
        return predicted_class


# Main function to classify an uploaded image
def classify(image, model_choice):
    # Load the model based on the selected model choice
    model = load_model(model_choice)

    # Preprocess the input image, which involves resizing, converting to tensor, and normalizing
    image_tensor = preprocess(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension as model expects a 4D tensor

    # Directory containing labeled subdirectories for each bird species
    data_dir = "data/Master"
    
    # List all subdirectory names, assuming each represents a class label
    class_names = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]

    # Predict the class of the image
    predicted_class = predict_image_class(model, image_tensor, class_names)
    return predicted_class


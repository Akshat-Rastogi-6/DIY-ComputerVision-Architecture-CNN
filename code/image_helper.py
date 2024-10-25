# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Developer details: 
        # Name: Akshat Rastogi, Rupal Mishra
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
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

# Function to resize an image to a specified size, here (150, 150)
def resize(img):
    # Define the resizing transformation with the target dimensions
    resize_transform = transforms.Resize((150, 150))
    
    # Apply the resizing transformation to the image
    img = resize_transform(img)
    
    return img


# Function to convert an image to a tensor format, with an option to display it
def to_tensor(img, show_image=True):
    # Convert the image to RGB format if it's not already in RGB, which standardizes the format
    img = img.convert("RGB")
    
    # Define the transformation to convert the image to a PyTorch tensor
    to_tensor_transform = transforms.ToTensor()
    
    # Apply the transformation to convert the image to a tensor
    img_tensor = to_tensor_transform(img)
    
    # Optionally display the image for visual verification
    if show_image:
        # Convert the tensor from [C, H, W] to [H, W, C] format for proper visualization
        np_img = img_tensor.permute(1, 2, 0).numpy()
        
        # Display the image using Matplotlib
        plt.imshow(np_img)
        plt.axis('off')  # Remove axis markings for a cleaner look
        
        # Save the displayed image to a file for future reference
        plt.savefig("code/saved_file/tensor.jpg")
    
    return img_tensor


# Function to normalize the image tensor to have zero mean and unit variance (common in CNNs)
def normalize(img):
    # Define the normalization transformation with mean and standard deviation values
    normalize_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    # Apply the normalization transformation to the image tensor
    img = normalize_transform(img)
    
    return img


# Function to ensure the image is in RGB format
def to_rgb(img):
    # Convert the image to RGB format if it isn't already, ensuring compatibility with certain models
    img = img.convert("RGB")
    
    return img


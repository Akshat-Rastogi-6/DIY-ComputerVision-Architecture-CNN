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
import os  # To manage directories and file paths
import torch  # PyTorch library for building and training models
import torch.nn as nn  # Neural network module in PyTorch
import torch.optim as optim  # Optimizer module in PyTorch
from torchvision import datasets, transforms  # For datasets and image transformations
from torch.utils.data import DataLoader  # To manage data batching and loading
from train import BirdClassificationCNN  # Import custom CNN model for bird classification

# Image dimensions
img_height, img_width = 150, 150 

# Function to load and preprocess data
def data_load(data_dir):
    """
    Load and preprocess the training and validation datasets with data augmentation.
    
    Parameters:
    - data_dir (str): Directory containing the dataset images.
    
    Returns:
    - train_loader (DataLoader): Data loader for the training dataset.
    - val_loader (DataLoader): Data loader for the validation dataset.
    """
    
    # Data transformations for training (data augmentation)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((img_height, img_width)),  # Random crop and resize
        transforms.RandomRotation(20),  # Random rotation up to 20 degrees
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.ToTensor(),  # Convert image to PyTorch tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    # Data transformations for validation/testing (no augmentation, standard resize)
    test_transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),  # Resize to fixed size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    # Load the datasets
    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(data_dir), transform=test_transform)

    # Data loaders to batch and shuffle data
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Training data loader
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # Validation data loader

    return train_loader, val_loader  # Return data loaders for training and validation sets

# Function to load the pretrained model
def load_model(model_path):
    """
    Load a pretrained model for bird classification from a saved state dictionary.
    
    Parameters:
    - model_path (str): Path to the saved model's state dictionary.
    
    Returns:
    - model (BirdClassificationCNN): Loaded model ready for evaluation or further training.
    """
    
    # Initialize the model with a specified number of classes (e.g., 25 bird species)
    model = BirdClassificationCNN(num_classes=25)
    
    # Load the model's state dictionary to retrieve saved weights and biases
    model.load_state_dict(torch.load(model_path))

    return model  # Return the loaded model

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

import numpy as np

import numpy as np

def validate_model(model, val_loader):
    correct = 0
    total = 0
    
    for inputs, labels in val_loader:
        # Convert PyTorch tensors to numpy arrays if using PyTorch DataLoader
        inputs = inputs.numpy()  # Assuming inputs are a PyTorch tensor
        labels = labels.numpy()  # Assuming labels are also in tensor form

        # Rearrange dimensions from (batch_size, channels, height, width) to (batch_size, height, width, channels)
        inputs = np.transpose(inputs, (0, 2, 3, 1))  # Change dimensions to (batch_size, height, width, channels)

        # Make predictions
        outputs = model.predict(inputs)  # Use Keras's predict method
        predicted = np.argmax(outputs, axis=1)  # Get the predicted class indices
        
        total += labels.size
        correct += np.sum(predicted == labels)  # Count correct predictions

    accuracy = 100 * correct / total
    return f'Validation Accuracy: {accuracy:.2f}%'

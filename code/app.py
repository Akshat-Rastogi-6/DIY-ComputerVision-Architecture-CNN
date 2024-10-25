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
import streamlit as st
import zipfile
from evaluate import validate_model
from classification import classify
from train import train_model
from load import data_load
from PIL import Image


st.set_page_config(page_title="Indian Bird Classification", page_icon=":cash:", layout="centered")
st.markdown("<h1 style='text-align: center; color: white;'>Indian Bird Classification</h1>", unsafe_allow_html=True)
st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["Model Config", "Model Training", "Model Evaluation", "Model Prediction"])

# import os
# import streamlit as st

# Define the default path
default_path = "data/Master"

with tab1:
    # Input field to take the directory path
    data_path = st.text_input("Enter the path to the folder containing images", value=default_path)
    extraction_dir = data_path
    if os.path.exists(data_path):
        # List all files in the directory and subdirectories
        file_list = [os.path.join(dp, f) for dp, dn, filenames in os.walk(data_path) for f in filenames]
        
        # Count the number of image files
        image_count = sum(1 for file_name in file_list if file_name.lower().endswith(('.jpeg', '.jpg')))
        
        # Display the total number of images found
        st.write(f"Number of images found: {image_count}")
        st.write(f"Images found in: {data_path}")
    else:
        st.write("The specified path does not exist. Please enter a valid path.")


with tab2:
    st.subheader("Model Training")
    st.write("This is where you can train the model.")
    st.divider()

    model_name = 'CNN'
    st.markdown(f"<h3 style='text-align: center; color: white;'>{model_name}</h3>", unsafe_allow_html=True)
    epochs = st.number_input('Number of Epochs:', min_value=1, max_value=100, value=10, step=1)

    if st.button(f"Train {model_name} Model", use_container_width=True):
        with st.status(f"Training {model_name} Model..."):
            train_loader, val_loader = data_load(extraction_dir)
            model, training_report = train_model(train_loader, val_loader, epochs)
            accuracy = validate_model(model, val_loader)
            for report in training_report:
                st.write(f"Training complete! -> {report}")

        st.success(f"{model_name} Trained Sucessully")

        st.write(f"Accuracy: {accuracy}")

    st.divider()

import tensorflow as tf
with tab3:
    st.subheader("Model Evaluation")
    st.write("This is where you can see the current metrics of the latest saved model.")
    st.divider()
    st.markdown(f"<h3 style='text-align: center; color: white;'> Classification Report </h3>", unsafe_allow_html=True)
    import traceback
    try:
        # Loading model
        model_path = r"code\saved_model\bird_classification_cnn.h5"
        st.write(f"Loading model from {model_path}")  # Debug: Print the model path
        model = tf.keras.models.load_model(model_path)
        st.write("Model loaded successfully!")  # Debug: Confirm the model is loaded

        # Load the validation data
        st.write("Loading validation data...")  # Debug: Add message before loading data
        train_loader, val_loader = data_load(extraction_dir)
        st.write(f"Validation data loaded. Total batches: {len(val_loader)}")  # Debug: Confirm data loaded

        # Validate the model
        st.write("Validating model...")  # Debug: Add message before validation
        cf = validate_model(model, val_loader)
        st.text(cf)

        st.divider()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")  # Basic error message
        # Print full traceback for detailed debugging
        st.error(traceback.format_exc())


with tab4:
    # Image path input instead of file uploader
    image_path = st.text_input("Enter the path of the image", value="data/Master/Common-Tailorbird/Common-Tailorbird_103.jpg")

    if image_path:
        try:
            # Open and display the image
            image = Image.open(image_path)
            # Display the predicted class
            predicted_class = classify(image_path)
            st.write(f"Predicted Class: {predicted_class}")
            # Show the image with an improved caption
            st.image(image, caption="Uploaded Image", use_column_width=True)
        except Exception as e:
            st.error(f"Error processing the image: {e}")
    else:
        st.write("No image path provided. Please enter the path of an image to classify.")





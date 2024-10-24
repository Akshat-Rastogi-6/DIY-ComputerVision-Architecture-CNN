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

with tab1:
    uploaded_file = st.file_uploader("Upload a ZIP file containing images in subfolders (JPEG/JPG format)", type="zip")

    if uploaded_file is not None:
        # Define the directory to extract to
        extraction_dir = "extracted_images"

        # Create the directory if it doesn't exist
        os.makedirs(extraction_dir, exist_ok=True)

        # Create a ZipFile object from the uploaded file
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            # Extract all contents to the specified directory
            zip_ref.extractall(extraction_dir)
            
            # Get the list of all files in the zip
            file_list = zip_ref.namelist()
            
            # Count the number of image files
            image_count = sum(1 for file_name in file_list if file_name.endswith(('.jpeg', '.jpg')))
        
        # Display the total number of images found
        st.write(f"Number of images found: {image_count}")
        st.write(f"Images have been extracted to: {extraction_dir}")
    else:
        st.write("No ZIP file uploaded yet.")


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


    with tab3:
        st.subheader("Model Evaluation")
        st.write("This is where you can see the current metrics of the latest saved model.")
        st.divider()
        st.markdown(f"<h3 style='text-align: center; color: white;'> Classification Report </h3>", unsafe_allow_html=True)
        try:
            cf = validate_model(model, val_loader)
            st.text(cf)
            st.divider()
        except Exception as e:
            st.error("Please upload a zip file.")

    with tab4:
    # Image uploader with improved title and description
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # try:
                # Open and display the uploaded image
                image = Image.open(uploaded_file)
                # Display the predicted class
                predicted_class = classify(uploaded_file)
                st.write(predicted_class)
                # Show the image with an improved caption
                st.image(image, caption="Uploaded Image", use_column_width=True)
            # except Exception as e:
            #     st.error(f"Error processing the image: {e}")
        else:
            st.write("No image uploaded yet. Please upload an image to classify.")





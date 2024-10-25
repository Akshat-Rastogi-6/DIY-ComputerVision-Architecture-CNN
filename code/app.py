# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Developer details: 
        # Name: Akshat Rastogi, Shubh Gupta and Rupal Mishra
        # Role: Developers
        # Code ownership rights: PreProd Corp
    # Version:
        # Version: V 1.1 (25 September 2024)
            # Developers: Akshat Rastogi, Shubh Gupta and Rupal Mishra
            # Unit test: Pass
            # Integration test: Pass
     
    # Description: This Streamlit app allows users to input features and make predictions using Unsupervised Learning.
        # Postgres: Yes
        # MQs: No
        # Cloud: No
        # Data versioning: No
        # Data masking: No

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Dependency: 
    # Environment:     
        # Python 3.11.5
        # Streamlit 1.36.0
# Importing necessary libraries for the application
import os  # To interact with the operating system for file path management
import streamlit as st  # Streamlit for building the web app
from evaluate import validate_model  # Importing the function to validate the model
from classification import classify  # Importing classification function (details not shown)
from image_helper import normalize, resize, to_rgb, to_tensor  # Importing image processing functions
from train import BirdClassificationCNN, train_model  # Importing model architecture and training function
from load import data_load, load_model  # Importing data loading and model loading functions
from PIL import Image  # PIL for image processing functionalities
from ingest_transform import store_data_path_in_postgresql, retrieve_data_path_from_postgresql  # PostgreSQL data handling
from ingest_transform_couchdb import store_data_path_in_couchdb, retrieve_data_path_from_couchdb  # CouchDB data handling

# Configuring the Streamlit app's page
st.set_page_config(page_title="Indian Bird Classification", page_icon=":cash:", layout="centered")  # Setting title, icon, and layout
st.markdown("<h1 style='text-align: center; color: white;'>Indian Bird Classification</h1>", unsafe_allow_html=True)  # Main title with styling
st.divider()  # Adding a horizontal divider

# Creating tabs for different functionalities of the app
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Model Config", "Model Training", "Model Evaluation", "Model Prediction", "Model Flow", "About"])  
default_path = "data/Master"  # Default path for images

# First tab: Image Folder Path Storage
with tab1:
    st.title("Image Folder Path Storage")  # Title for the first tab

    # Input field to take the directory path for image storage
    data_path = st.text_input("Enter the path to the folder containing images", value=default_path)  # User input for directory
    # Dropdown to select the database for storing the data path
    database_choice = st.selectbox("Select the database to store the data path:", ("PostgreSQL", "CouchDB"))  

    # Check if the provided path exists
    if os.path.exists(data_path):
        # List all files in the specified directory and subdirectories
        file_list = [os.path.join(dp, f) for dp, dn, filenames in os.walk(data_path) for f in filenames]
        
        # Count the number of image files (with .jpeg and .jpg extensions)
        image_count = sum(1 for file_name in file_list if file_name.lower().endswith(('.jpeg', '.jpg')))
        
        # Display the total number of images found
        st.write(f"Number of images found: {image_count}")  # Displaying count
        st.write(f"Images found in: {data_path}")  # Displaying the path where images were found
        
        # Button to store the data path in the selected database
        if st.button("Store Data Path"):
            # Based on user's choice, store data path in the respective database
            if database_choice == "PostgreSQL":
                store_data_path_in_postgresql(data_path)  # Function to store path in PostgreSQL
            elif database_choice == "CouchDB":
                store_data_path_in_couchdb(data_path)  # Function to store path in CouchDB
    else:
        # Message if the specified path does not exist
        st.write("The specified path does not exist. Please enter a valid path.")  # Alert for invalid path


# Importing additional functionalities related to ResNet
import streamlit as st  # Streamlit for building the web app
from load_resnet import create_pretrained_resnet  # Importing function to create a pretrained ResNet model

# Second tab: Model Training
with tab2:
    st.subheader("Model Training")  # Subheader for the Model Training section
    st.write("This is where you can train the model.")  # Brief description of the section
    st.divider()  # Adding a horizontal divider for visual separation

    # Setting the model name for display purposes
    model_name = 'CNN'  # The name of the model being trained
    # Displaying the model name as a header
    st.markdown(f"<h3 style='text-align: center; color: white;'>{model_name}</h3>", unsafe_allow_html=True)
    
    # Input field for the user to specify the number of training epochs
    epochs = st.number_input('Number of Epochs:', min_value=1, max_value=100, value=10, step=1)  
    # The user can select a number between 1 and 100, defaulting to 10 epochs

    # Placeholder for database choice if needed
    # This section could be expanded later if more database functionalities are introduced

    # Button to initiate the model training process
    if st.button(f"Train {model_name} Model", use_container_width=True):
        # Displaying status message while training
        with st.status(f"Training {model_name} Model..."):
            # Retrieving the extraction directory based on the selected database
            # Use PostgreSQL or CouchDB to retrieve the path where images are stored
            extraction_dir = retrieve_data_path_from_postgresql() if database_choice == "PostgreSQL" else retrieve_data_path_from_couchdb()
            
            # Loading the training and validation data using the data_load function
            train_loader, val_loader = data_load(extraction_dir)
            
            # Training the model with the loaded data, number of epochs, and database choice
            model, training_report = train_model(train_loader, val_loader, epochs, database_choice)
            
            # Validating the model's performance on the validation dataset
            accuracy = validate_model(model, val_loader)
            
            # Displaying the training report details after training completion
            for report in training_report:
                st.write(f"Training complete! -> {report}")

        # Displaying success message once the model is trained
        st.success(f"{model_name} Trained Successfully")
        # Showing the accuracy of the trained model
        st.write(f"Accuracy: {accuracy}")

    st.divider()  # Divider between sections for better readability

    # Pretrained model section
    st.subheader("Load Pretrained Model")  # Subheader for loading pretrained models
    st.write("You can load a pretrained ResNet model below.")  # Brief description
    st.divider()  # Adding a divider for visual separation

    # Input field for the user to specify the number of classes in the pretrained model
    num_classes = st.number_input('Number of Classes for Pretrained Model:', min_value=1, max_value=100, value=25, step=1)  
    # The user can select a number between 1 and 100, defaulting to 25 classes

    # Button to load the pretrained ResNet model
    if st.button("Load Pretrained ResNet Model", use_container_width=True):
        # Displaying status message while loading the model
        with st.status("Loading pretrained model..."):
            try:
                # Calling the function to create and save the pretrained ResNet model
                model = create_pretrained_resnet(num_classes)
                # Success message once the model is loaded
                st.success("Pretrained ResNet model loaded successfully.")
            except Exception as e:
                # Displaying error message if there is an issue while loading the model
                st.error(f"Error loading model: {e}")

    st.divider()  # Optional: additional divider for clarity


# Third tab: Model Evaluation
with tab3:
    st.subheader("Model Evaluation")  # Subheader for the Model Evaluation section
    st.write("This is where you can see the current metrics of the latest saved model.")  # Brief description
    st.divider()  # Adding a horizontal divider for visual separation
    
    # Displaying the title for the classification report
    st.markdown(f"<h3 style='text-align: center; color: white;'> Classification Report </h3>", unsafe_allow_html=True)

    try:
        # Loading the trained model from a specified file path
        model_path = r"code\saved_model\bird_classification_cnn.pth"  # Path to the saved model file
        st.write(f"Loading model from {model_path}")  # Debug message: Print the model path being loaded
        
        # Call the function to load the model
        model = load_model(model_path)  # Function defined elsewhere to load the model
        model.eval()  # Set the model to evaluation mode (disables dropout, batch normalization, etc.)
        
        # Confirm successful model loading
        st.write("Model loaded successfully!")  # Debug message to confirm the model has been loaded

        # Load the validation data for evaluation
        st.write("Loading validation data...")  # Debug message: Informing that validation data is being loaded
        train_loader, val_loader = data_load(default_path)  # Load the data, assuming it returns loaders for training and validation
        
        # Confirm the number of batches in the validation dataset
        st.write(f"Validation data loaded. Total batches: {len(val_loader)}")  # Debug message: Confirm data loaded

        # Validate the model using the validation data
        st.write("Validating model...")  # Debug message: Indicating the start of model validation
        cf = validate_model(model, val_loader)  # Call the validation function, returning the confusion matrix or classification report
        st.text(cf)  # Display the classification report or metrics in text format

        st.divider()  # Divider to separate different sections of the app
    except Exception as e:
        # Error handling: if any exceptions occur, display an error message
        st.error(f"An error occurred: {str(e)}")  # Displaying the error message



    

# Fourth tab: Model Prediction
with tab4:
    # Image uploader with an improved title and description for user guidance
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])  # Allow users to upload image files
    st.write("Choose Model for Prediction:")  # Instruction for the user regarding model selection

    # Dropdown for selecting the model to use for prediction
    model_choice = st.selectbox("Model", ["Self-trained CNN", "Pretrained ResNet"])  # User can select between two model options

    # Check if a file has been uploaded
    if uploaded_file is not None:
        # If an image is uploaded, open it using PIL
        image = Image.open(uploaded_file)  # Open the uploaded image file
        
        # Perform classification using the selected model and the uploaded image
        predicted_class = classify(uploaded_file, model_choice)  # Call the classify function to get the prediction
        st.write(f"Predicted Class: {predicted_class}")  # Display the predicted class to the user
        
        # Show the uploaded image with an improved caption
        st.image(image, caption="Uploaded Image", use_column_width=True)  # Display the image with a caption and responsive width
    else:
        # If no image has been uploaded, prompt the user
        st.write("No image uploaded yet. Please upload an image to classify.")  # Message to guide the user to upload an image

# Fifth tab: Image Preprocessing
with tab5:
    # File uploader for the user to upload an image file
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="second")
    
    # Check if the user has uploaded a file
    if uploaded_file is not None:
        # Open the uploaded image using PIL
        image = Image.open(uploaded_file)
        
        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.divider()

        # Section for image resizing
        st.markdown(f"<h4 style='text-align: center; color: white;'>Image Resizing</h4>", unsafe_allow_html=True)
        st.write("Resizing the image ensures that all images have the same dimensions (img_height, img_width) so they can be processed in batches and passed through the network consistently. This also reduces computational complexity when using large images.")
        img = resize(image)  # Resize the image to specified dimensions
        st.image(img, caption="Resized Image")  # Display the resized image
        st.divider()

        # Section for converting the image to RGB
        st.markdown(f"<h4 style='text-align: center; color: white;'>Convert to RGB</h4>", unsafe_allow_html=True)
        st.write("Some images may have different formats (e.g., grayscale, RGBA). The conversion to RGB ensures consistency by transforming all images into 3-channel RGB format, which is necessary if you're using models that expect color images. This step prevents potential errors during processing.")
        img = to_rgb(img)  # Convert the image to RGB format
        st.image(img)  # Display the RGB image
        st.divider()

        # Section for converting the image to tensor format
        st.markdown(f"<h4 style='text-align: center; color: white;'>Convert Image to Tensor</h4>", unsafe_allow_html=True)
        st.write("Machine learning models, especially in PyTorch, expect data in tensor format. This transformation converts the image from a PIL image (or NumPy array) to a PyTorch tensor and scales the pixel values to the range [0, 1], which is better suited for numerical computation.")
        tensor = to_tensor(img)  # Convert the image to tensor
        st.write(tensor)  # Display the tensor (this might display a lot of data, consider formatting)
        st.image("code/saved_file/tensor.jpg")  # Show a sample tensor image if available
        st.divider()

        # Section for normalizing the tensor
        st.markdown(f"<h4 style='text-align: center; color: white;'>Normalizing</h4>", unsafe_allow_html=True)
        st.write("Normalization adjusts the pixel values in each channel (RGB) to have zero mean and unit variance based on the specific mean and standard deviation. This helps stabilize and speed up training by ensuring that the inputs to the neural network are on a consistent scale, leading to faster convergence. The values [0.485, 0.456, 0.406] and [0.229, 0.224, 0.225] are commonly used for datasets like ImageNet to match pre-trained model expectations.")
        normalized_tensor = normalize(tensor)  # Normalize the tensor
        st.write(normalized_tensor)  # Display the normalized tensor
        st.divider()

        # Adding a batch dimension to the tensor
        st.write("After that, we add a batch dimension to the tensor.")
        batch_normalized_tensor = normalized_tensor.unsqueeze(0)  # Add a batch dimension
        st.write(batch_normalized_tensor)  # Display the batch-normalized tensor

        # Final message about model processing
        st.markdown(f"<h3 style='text-align: center; color: white;'>After all this pre-processing, the model is passed through and classification is done.</h3>", unsafe_allow_html=True)
    else:
        # Prompt for no uploaded image
        st.write("No image uploaded yet. Please upload an image to classify.")

# Sixth tab: About Indian Bird Classification
with tab6:
    st.title("About Indian Bird Classification")

    # Overview of the website
    st.markdown(""" 
    ### Website Overview
    The **Indian Bird Classification** platform is designed to classify bird species found in India using machine learning techniques. 
    The website allows users to upload datasets, train models, evaluate their performance, and make predictions on bird species.
    
    The goal of this platform is to provide an easy-to-use interface for ornithologists, researchers, and hobbyists 
    who are interested in identifying bird species using images.
    """)
    st.markdown("---")

    # Introduction to CNNs
    st.header("What is a Convolutional Neural Network (CNN)?")
    st.markdown(""" 
    A **Convolutional Neural Network (CNN)** is a type of deep learning algorithm primarily used for image recognition 
    and classification tasks. CNNs are designed to automatically and adaptively learn spatial hierarchies of features 
    from input images.

    CNNs consist of several layers:
    - **Convolutional Layers:** Apply filters (kernels) to the input image to extract features like edges, textures, etc.
    - **Pooling Layers:** Reduce the spatial dimensions (width, height) of the input to make computation more efficient.
    - **Fully Connected Layers:** Connect all neurons from the previous layer to every neuron in the next layer, similar to traditional neural networks.
    """)

    # Display basic CNN architecture image
    st.image(r"code\images\cover.png", caption="Basic CNN Architecture", use_column_width=True)

    # Expandable section for more information about CNN layers
    with st.expander("Learn more about CNN Layers"):
        st.markdown("""
        - **Convolutional Layer:** This is the core building block of a CNN. It performs convolution operations on the input data with a kernel or filter. 
        This layer helps in detecting features like edges, corners, and textures in images.
        
        - **Activation Function (ReLU):** After each convolution operation, an activation function (typically ReLU) is applied, introducing non-linearity into the model.
        
        - **Pooling Layer:** After the convolutional layer, pooling is applied to reduce the dimensions and extract dominant features. This helps in reducing the number of parameters and computation in the network.
        
        - **Fully Connected Layer:** In the final stages of CNN, the fully connected layer classifies the image into the target categories.
        """)

    st.markdown("---")

    # Further reading resources
    st.subheader("Further Reading")
    st.markdown(""" 
    If you are interested in learning more about CNNs or machine learning in general, here are a few useful resources:

    - [Deep Learning with Python by François Chollet](https://www.manning.com/books/deep-learning-with-python)
    - [Stanford University's CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
    - [Machine Learning on Coursera](https://www.coursera.org/learn/machine-learning)
    """)


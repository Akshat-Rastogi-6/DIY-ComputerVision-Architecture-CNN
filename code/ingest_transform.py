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

# Import necessary libraries
import torch  # PyTorch for model handling
import torch.nn as nn  # Neural network module in PyTorch
from torchvision import transforms  # For preprocessing image data
from PIL import Image  # Image processing
import psycopg2  # PostgreSQL database connector
import streamlit as st  # Web app framework

# Define image dimensions for preprocessing
img_height, img_width = 150, 150 

# Preprocess function to apply transformations to an input image
def preprocess(img_path):
    """
    Preprocess the input image by resizing, normalizing, and converting it to a tensor format.
    
    Parameters:
    - img_path (str): The file path to the image.

    Returns:
    - img (Tensor): The transformed image tensor ready for model input.
    """
    # Define the sequence of transformations: resize, convert to tensor, normalize
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),  # Resize image to fixed dimensions
        transforms.ToTensor(),  # Convert image to tensor format
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize using ImageNet means/stds
    ])
    
    # Open the image using PIL, convert it to RGB (in case it is grayscale), and apply transformations
    img = Image.open(img_path)
    img = img.convert("RGB")  # Ensure image is in RGB format for model compatibility
    img = transform(img)
    return img

import psycopg2
import streamlit as st

# PostgreSQL database connection function
def connect_postgresql():
    """
    Establish a connection to the PostgreSQL database.
    
    Returns:
    - conn (Connection object or None): PostgreSQL connection object if successful; None otherwise.
    """
    try:
        # Connect to the PostgreSQL database with host, database, user, and password details
        conn = psycopg2.connect(
            host="localhost",
            database="indian_bird",
            user="postgres",
            password="123456"
        )
        return conn
    except Exception as e:
        # Show an error message on the Streamlit app if connection fails
        st.write(f"PostgreSQL connection error: {e}")
        return None

# Function to store a data path in the PostgreSQL database
def store_data_path_in_postgresql(data_path):
    """
    Store a file path in the PostgreSQL table 'data_paths'.
    
    Parameters:
    - data_path (str): The file path of the data to be stored.
    """
    conn = connect_postgresql()
    if conn is not None:
        try:
            conn.autocommit = True  
            cur = conn.cursor()  # Create a cursor for executing queries
            
            # Create the table if it doesn't exist already
            cur.execute("""
                CREATE TABLE IF NOT EXISTS data_paths (
                    id SERIAL PRIMARY KEY,  -- Unique ID that auto-increments for each entry
                    path TEXT NOT NULL      -- Column to store the data path, cannot be NULL
                )
            """)
            
            # Insert the data path into the table
            cur.execute("INSERT INTO data_paths (path) VALUES (%s)", (data_path,))
            conn.commit()  # Commit the transaction
            st.write("Data path stored in PostgreSQL successfully.")
        except Exception as e:
            # Show an error message on the Streamlit app if insertion fails
            st.write(f"Error storing data path in PostgreSQL: {e}")
        finally:
            # Always close the cursor and the connection, whether successful or not
            if cur:
                cur.close()
            conn.close()

# Function to store a model path in the PostgreSQL database
def store_model_path_in_postgresql(model_path):
    """
    Store a model file path in the PostgreSQL table 'model_paths'.
    
    Parameters:
    - model_path (str): The file path of the model to be stored.
    """
    conn = connect_postgresql()
    if conn is not None:
        try:
            conn.autocommit = True  # Automatically commit changes
            cur = conn.cursor()  # Cursor to interact with the database
            
            # Create the 'model_paths' table if it doesn't already exist
            cur.execute("""
                CREATE TABLE IF NOT EXISTS model_paths (
                    id SERIAL PRIMARY KEY,  -- Unique identifier that increments automatically
                    path TEXT NOT NULL      -- Column to store the model path, cannot be NULL
                )
            """)
            
            # Insert the model path into the 'model_paths' table
            cur.execute("INSERT INTO model_paths (path) VALUES (%s)", (model_path,))
            conn.commit()  # Commit the transaction
            st.write("Model path stored in PostgreSQL successfully.")
        except Exception as e:
            # Display an error message if insertion fails
            st.write(f"Error storing model path in PostgreSQL: {e}")
        finally:
            # Always close the cursor and the connection after use
            if cur:
                cur.close()
            conn.close()

# Function to retrieve the most recent data path stored in PostgreSQL
def retrieve_data_path_from_postgresql():
    """
    Retrieve the latest data path stored in the PostgreSQL 'data_paths' table.
    
    Returns:
    - data_path (str or None): The most recent data path if available; None otherwise.
    """
    conn = connect_postgresql()
    if conn is not None:
        try:
            cur = conn.cursor()  # Cursor to execute database queries
            
            # Query the table to get the latest data path (based on highest ID)
            cur.execute("SELECT path FROM data_paths ORDER BY id DESC LIMIT 1")
            result = cur.fetchone()  # Fetch the first result from the query
            
            # Check if a result is found
            if result:
                data_path = result[0]  # Extract the data path from the result
                st.write(f"Retrieved data path from PostgreSQL: {data_path}")
                return data_path
            else:
                # Inform the user if there is no data path in the table
                st.write("No data path found in PostgreSQL.")
                return None
        except Exception as e:
            # Display an error message if retrieval fails
            st.write(f"Error retrieving data path from PostgreSQL: {e}")
            return None
        finally:
            # Close the cursor and connection to free up resources
            if cur:
                cur.close()
            conn.close()

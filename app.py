import streamlit as st
import numpy as np
import os
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.initializers import Orthogonal, GlorotUniform, Zeros
from tensorflow.keras.preprocessing import image
from keras.preprocessing.sequence import pad_sequences
import dill

# Ensure correct Keras initializers are used
initializers = {
    'Orthogonal': Orthogonal(),
    'GlorotUniform': GlorotUniform(),
    'Zeros': Zeros()
}

# Function to load models safely
def safe_load_model(model_path):
    try:
        model = load_model(model_path, custom_objects=initializers)
        st.write(f"Loaded model from {model_path}")
        return model
    except Exception as e:
        st.error(f"Failed to load model from {model_path}: {e}")
        return None

# Function to load model from JSON safely
def safe_load_model_from_json(json_path, weights_path):
    try:
        with open(json_path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects=initializers)
        model.load_weights(weights_path)
        st.write(f"Loaded model from {json_path} and {weights_path}")
        return model
    except Exception as e:
        st.error(f"Failed to load model from {json_path} and {weights_path}: {e}")
        return None

# Define the base path to your files
base_path = os.path.dirname(os.path.abspath(__file__))

# Load the trained models
lstm_model = safe_load_model(os.path.join(base_path, 'LSTM_model.h5'))
cnn_model = safe_load_model_from_json(os.path.join(base_path, 'CNN_MODEL.json'), os.path.join(base_path, 'CNN_MODEL_weights.h5'))
meta_model = safe_load_model(os.path.join(base_path, 'Meta_Model.save.h5'))

# Load additional data needed for predictions
try:
    with open(os.path.join(base_path, 'autism_data.pkl'), 'rb') as f:
        encoded_sequences, categorical_cols, max_sequence_length, X_train_cat = dill.load(f)
    st.write("Loaded additional data for predictions")
except Exception as e:
    st.error(f"Failed to load additional data: {e}")

# Function to predict with LSTM model
def predict_with_lstm(sequence, categorical_features):
    padded_sequence = pad_sequences([sequence], maxlen=max_sequence_length, padding='post', truncating='post')
    padded_sequence_reshaped = np.expand_dims(padded_sequence, axis=-1)
    categorical_features_int = np.array([categorical_features]).astype(int)
    prediction = lstm_model.predict([padded_sequence_reshaped, categorical_features_int])
    return prediction[0][0]

# Function to predict with CNN model
img_width, img_height = 150, 150

def predict_with_cnn(image_path):
    img = image.load_img(image_path, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x /= 255.0  # Rescale if necessary
    x = np.expand_dims(x, axis=0)  # Add batch dimension
    prediction = cnn_model.predict(x)
    return prediction[0][0]

# Streamlit interface
st.title("Autism Prediction Ensemble Model")

# Input fields for MCHAT details
st.header("Enter MCHAT Details")
sequence = []
for i in range(1, 11):
    sequence.append(st.selectbox(f"A{i}", [0, 1]))

# Input fields for categorical features
categorical_features = []
for col in X_train_cat.columns:
    categorical_features.append(st.selectbox(col, [0, 1]))

# File uploader for image
st.header("Upload Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if st.button("Predict"):
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_image_path = os.path.join(base_path, "temp_image.jpg")
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Get predictions
        if lstm_model and cnn_model and meta_model:
            lstm_prediction = predict_with_lstm(sequence, categorical_features)
            cnn_prediction = predict_with_cnn(temp_image_path)
            ensemble_prediction = meta_model.predict([np.array([lstm_prediction]), np.array([cnn_prediction])])
        
            # Convert the prediction to class label (0 or 1)
            predicted_class = 1 if ensemble_prediction > 0.5 else 0
            if predicted_class == 1:
                st.write("Prediction: Autistic")
            else:
                st.write("Prediction: Non-Autistic")
            
            st.write(f"Predicted Class (Ensemble): {predicted_class}")
        else:
            st.error("One or more models failed to load. Please check the logs for details.")
    else:
        st.write("Please upload an image file.")

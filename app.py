import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.preprocessing import image
from keras.preprocessing.sequence import pad_sequences
import dill

# Load the trained models
lstm_model = load_model('LSTM_model.h5')

# Load the trained CNN model from JSON
with open('CNN_MODEL.json', 'r') as json_file:
    cnn_model_json = json_file.read()
cnn_model = model_from_json(cnn_model_json)
cnn_model.load_weights('CNN_MODEL_WEIGHTS.h5')

meta_model = load_model('Meta_Model.h5')

# Load additional data needed for predictions
with open('autism_data.pkl', 'rb') as f:
    encoded_sequences, categorical_cols, max_sequence_length, X_train_cat = dill.load(f)

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
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Get predictions
        lstm_prediction = predict_with_lstm(sequence, categorical_features)
        cnn_prediction = predict_with_cnn("temp_image.jpg")
        ensemble_prediction = meta_model.predict([np.array([lstm_prediction]), np.array([cnn_prediction])])
        
        # Convert the prediction to class label (0 or 1)
        predicted_class = 1 if ensemble_prediction > 0.5 else 0
        if predicted_class == 1:
            st.write("Prediction: Autistic")
        else:
            st.write("Prediction: Non-Autistic")
        
        st.write(f"Predicted Class (Ensemble): {predicted_class}")
    else:
        st.write("Please upload an image file.")

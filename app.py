import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.preprocessing import image
from keras.preprocessing.sequence import pad_sequences
import dill
import os

# Function to load models safely
def safe_load_model(model_path):
    try:
        model = load_model(model_path, custom_objects={
            'Orthogonal': tf.keras.initializers.Orthogonal(),
            'GlorotUniform': tf.keras.initializers.GlorotUniform(),
            'Zeros': tf.keras.initializers.Zeros()
        })
        return model
    except Exception as e:
        st.error(f"Failed to load model from {model_path}: {e}")
        return None

# Function to load model from JSON safely
def safe_load_model_from_json(json_path, weights_path):
    try:
        with open(json_path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json)
        model.load_weights(weights_path)
        return model
    except Exception as e:
        st.error(f"Failed to load model from {json_path} and {weights_path}: {e}")
        return None

# Define the base path to your files
base_path = os.path.dirname(os.path.abspath(__file__))

# Load models
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

# Prediction functions
def predict_with_lstm(sequence, categorical_features):
    padded_sequence = pad_sequences([sequence], maxlen=max_sequence_length, padding='post', truncating='post')
    padded_sequence_reshaped = np.expand_dims(padded_sequence, axis=-1)
    categorical_features_int = np.array([categorical_features]).astype(int)
    prediction = lstm_model.predict([padded_sequence_reshaped, categorical_features_int])
    return prediction[0][0]

def predict_with_cnn(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x /= 255.0  # Rescale if necessary
    x = np.expand_dims(x, axis=0)  # Add batch dimension
    prediction = cnn_model.predict(x)
    return prediction[0][0]

# Define page functions
def home():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("img/B1.jpg");
            background-size: cover;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("Home Page")
    st.write("Welcome to the Autism Prediction Ensemble Model application!")
def predict():
    st.title("Autism Prediction Ensemble Model")

    col1, col2 = st.columns(2)

    A_questions = [
        "Does your child look at you when you call his/her name?",
        "How easy is it for you to get eye contact with your child?",
        "Does your child point to indicate that s/he wants something? (e.g. a toy that is out of reach)",
        "Does your child point to share interest with you? (e.g. pointing at an interesting sight)",
        "Does your child pretend? (e.g. care for dolls, talk on a toy phone)",
        "Does your child follow where you’re looking?",
        "If you or someone else in the family is visibly upset, does your child show signs of wanting to comfort them? (e.g. stroking hair, hugging them)",
        "Would you describe your child’s first words as:",
        "Does your child use simple gestures? (e.g. wave goodbye)",
        "Does your child stare at nothing with no apparent purpose?"
    ]

    with col1:
        st.header("ANSWER THE FOLLOWING QUESTIONS")
        sequence = [st.selectbox(f"Q{i+1} : {A_questions[i]}", [0, 1]) for i in range(10)]

    with col2:
        st.header("Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if st.button("Predict"):
        if uploaded_file is not None:
            temp_image_path = os.path.join(base_path, "temp_image.jpg")
            with open(temp_image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            if lstm_model and cnn_model and meta_model:
                with st.spinner('Predicting...'):
                    lstm_prediction = predict_with_lstm(sequence, categorical_features)
                    cnn_prediction = predict_with_cnn(temp_image_path)
                    ensemble_prediction = meta_model.predict([np.array([lstm_prediction]), np.array([cnn_prediction])])

                predicted_class = 1 if ensemble_prediction > 0.5 else 0
                st.success("Prediction: Autistic" if predicted_class == 1 else "Prediction: Non-Autistic")
                st.write(f"Predicted Class (Ensemble): {predicted_class}")
            else:
                st.error("One or more models failed to load. Please check the logs for details.")
        else:
            st.warning("Please upload an image file.")

def contact():
    st.title("Contact")
    st.write("You can reach us at contact@autism-prediction.com")

def info():
    st.title("Information")
    st.write("This application uses machine learning models to predict the likelihood of autism in children based on MCHAT details and images.")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict", "Contact", "Info"])

# Render the chosen page
if page == "Home":
    home()
elif page == "Predict":
    predict()
elif page == "Contact":
    contact()
elif page == "Info":
    info()

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Input, Concatenate, Dense
from tensorflow.keras.models import Model
import tenseal as ts
import os
import dill
from keras.preprocessing.sequence import pad_sequences

# Encryption setup
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,  # Consider increasing this if necessary
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.global_scale = 2**40
context.generate_galois_keys()

# Load models
base_path = os.path.dirname(os.path.abspath(__file__))
lstm_model = load_model(os.path.join(base_path, 'LSTM_model.h5'))
cnn_model = load_model(os.path.join(base_path, 'model.keras'))

# Create and compile the meta-model
lstm_input = Input(shape=(1,), name='lstm_input')
cnn_input = Input(shape=(1,), name='cnn_input')
concatenated = Concatenate()([lstm_input, cnn_input])
dense = Dense(64, activation='relu')(concatenated)
output = Dense(1, activation='sigmoid')(dense)
meta_model = Model(inputs=[lstm_input, cnn_input], outputs=output)
meta_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Encryption and decryption functions
def encrypt_data(data):
    return ts.ckks_vector(context, np.array(data).flatten())

def decrypt_data(encrypted_data):
    return np.array(encrypted_data.decrypt())

# Prediction functions
@tf.function
def lstm_predict(sequence, categorical):
    return lstm_model([sequence, categorical])

@tf.function
def cnn_predict(image):
    return cnn_model(image)

@tf.function
def meta_predict(lstm_output, cnn_output):
    return meta_model([lstm_output, cnn_output])

def predict_with_encrypted_lstm(encrypted_sequence, encrypted_categorical):
    decrypted_sequence = np.array(decrypt_data(encrypted_sequence)).reshape((1, 10, 1))
    decrypted_categorical = np.array(decrypt_data(encrypted_categorical)).reshape((1, 24))
    return lstm_predict(decrypted_sequence, decrypted_categorical).numpy()

def predict_with_encrypted_cnn(encrypted_image):
    decrypted_image = np.array(decrypt_data(encrypted_image)).reshape((1, 150, 150, 3))
    return cnn_predict(decrypted_image).numpy()

# Input validation
def validate_inputs(sequence, categorical, img_array):
    assert len(sequence) == 10, "Sequence must have 10 elements"
    assert len(categorical) == 24, "Categorical features must have 24 elements"
    assert img_array.shape == (150, 150, 3), "Image must be 150x150x3"

# Main prediction function
def predict_autism(sequence, categorical_features, image_path):
    # Load and preprocess image
    img = image.load_img(image_path, target_size=(150, 150))  # Resize to the expected input shape
    img_array = image.img_to_array(img) / 255.0  # Normalize

    # Validate inputs
    validate_inputs(sequence, categorical_features, img_array)

    # Encrypt inputs
    encrypted_sequence = encrypt_data(sequence)
    encrypted_categorical = encrypt_data(categorical_features)
    encrypted_image = encrypt_data(img_array)

    # Get predictions
    lstm_prediction = predict_with_encrypted_lstm(encrypted_sequence, encrypted_categorical)
    cnn_prediction = predict_with_encrypted_cnn(encrypted_image)

    # Make final prediction using meta-model
    final_prediction = meta_predict(
        tf.convert_to_tensor(lstm_prediction, dtype=tf.float32),
        tf.convert_to_tensor(cnn_prediction, dtype=tf.float32)
    ).numpy()

    predicted_class = 1 if final_prediction > 0.5 else 0
    return predicted_class, final_prediction.item()

# Streamlit Interface
def home():
    st.title("Home")
    st.header("Welcome to the Early Autism Prediction in Children App")
    st.write("This application predicts autism likelihood using encrypted data.")

def predict():
    st.title("Autism Prediction App (Encrypted Data)")

    # Example Questions and Image Upload Interface
    A_questions = [
        "Does your child look at you when you call his/her name?",
        "How easy is it for you to get eye contact with your child?",
        "Does your child point to indicate that s/he wants something? (e.g. a toy that is out of reach)",
        "Does your child point to share interest with you? (e.g. pointing at an interesting sight)",
        "Does your child pretend? (e.g. care for dolls, talk on a toy phone)",
        "Does your child follow where you’re looking?",
        "If you or someone else in the family is visibly upset, does your child show signs of wanting to comfort them? (e.g. stroking hair, hugging them)",
        "Would you describe your child’s first words as Normal:",
        "Does your child use simple gestures? (e.g. wave goodbye)",
        "Does your child stare at nothing with no apparent purpose?"
    ]

    options = {"No": 0, "Yes": 1}
    sequence = [options[st.selectbox(f"Q{i+1}: {A_questions[i]}", options.keys())] for i in range(10)]
    categorical_features = [0] * 24  # Placeholder for categorical features

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    consent = st.checkbox("I consent to the privacy policy and the processing of the uploaded image for prediction purposes.")

    if st.button("Predict") and consent:
        if uploaded_file is not None:
            # Save the uploaded image temporarily
            temp_image_path = os.path.join(base_path, "temp_image.jpg")
            with open(temp_image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                predicted_class, prediction_probability = predict_autism(sequence, categorical_features, temp_image_path)
                st.success("Prediction: Autistic" if predicted_class == 1 else "Prediction: Non-Autistic")
                st.write(f"Prediction Probability: {prediction_probability:.4f}")
            except AssertionError as e:
                st.error(f"Input validation failed: {e}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
            finally:
                # Delete the temporary image file
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
        else:
            st.warning("Please upload an image file.")
    elif not consent:
        st.warning("Please consent to the privacy policy before proceeding.")

def contact():
    st.title("Contact")
    st.write("You can reach us at contact@autism-prediction.com")

def info():
    st.title("Information")
    st.write("This application uses machine learning models and encryption to predict the likelihood of autism in children.")

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

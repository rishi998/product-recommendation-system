import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logging.basicConfig(level=logging.INFO)

MODEL_PATH = "trained_model_5000.keras"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    st.error("Failed to load model.")

JSON_PATH = "features_to_url.json"
try:
    with open(JSON_PATH, "r") as file:
        feature_to_url = json.load(file)
        labels = list(feature_to_url.keys()) 
    logging.info("JSON file loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load JSON file: {e}")
    st.error("Failed to load JSON data.")

vectorizer = TfidfVectorizer()
label_vectors = vectorizer.fit_transform(labels)

def preprocess_image(image):
    try:
        img = image.resize((224, 224))  
        img_array = np.array(img) / 255.0  
        img_array = np.expand_dims(img_array, axis=0)  
        return img_array
    except Exception as e:
        logging.error(f"Error in image preprocessing: {e}")
        st.error("Error in processing the image.")

def predict_label(image):
    try:
        preprocessed_image = preprocess_image(image)
        predictions = model.predict(preprocessed_image)
        predicted_label_index = np.argmax(predictions)
        predicted_label = labels[predicted_label_index]
        logging.info(f"Predicted Label: {predicted_label}")
        return predicted_label
    except Exception as e:
        logging.error(f"Error predicting label: {e}")
        st.error("Failed to predict label.")
        return None
    
st.title("Product-Recommender-System")

st.markdown("""
<style>
body {
    margin: 10px;
    border: 5px solid #f4f4f4;
    border-radius: 10px;
    padding:0px;
}
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])
with col2:
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=150) 

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    predicted_label = predict_label(image)

    query_vector = vectorizer.transform([predicted_label])
    similarities = cosine_similarity(query_vector, label_vectors).flatten()
    top_indices = np.argsort(similarities)[-5:][::-1]

    st.write("### Top 5 Similar Labels and Images:")
    for index in top_indices:
        label = labels[index]
        url = feature_to_url[label]
        if url:
            st.image(url, caption=f"Image for {label}")
            st.markdown(f"[{label} URL]({url})") 
        else:
            st.write(f"No images found for {label}")

import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="Crop Disease Detector", page_icon="🌱")
st.title("🌱 Intelligent Crop Disease Detection")
st.write("Upload a photo of a plant leaf, and the AI will diagnose it and provide agricultural advice.")

# 2. Load Model and Class Names (This runs once and caches for speed)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('plant_disease_model.h5')
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_model()

# 3. Advisory Database
advisory_db = {
    "Apple___Apple_scab": "Apply fungicides early in the season. Rake and destroy fallen leaves.",
    "Apple___Black_rot": "Prune out infected branches and remove mummified fruit. Apply a copper-based fungicide.",
    "Tomato___Late_blight": "This spreads fast! Remove and destroy infected plants immediately. Apply copper fungicide.",
    "Tomato___healthy": "Your plant is healthy! Maintain good spacing.",
    "Potato___Early_blight": "Apply a protective fungicide. Avoid wetting the leaves.",
    # Add more if you like!
}

# 4. UI: File Uploader
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    with st.spinner('Analyzing image...'):
        # Preprocess the image
        img = image.resize((224, 224))
        img_array = np.array(img)
        
        # Handle images with 4 channels (RGBA)
        if img_array.shape[-1] == 4:
            img_array = img_array[..., :3]
            
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
       predictions = model.predict(img_array)
       predicted_class = class_names[np.argmax(predictions)]
       confidence = np.max(predictions) * 100

# Check if the AI is confident enough
      if confidence < 85.0:
             st.error("⚠️ Image not recognized. Please upload a clear, close-up photo of a plant leaf.")
     else:
    # Format output
    clean_name = predicted_class.replace('___', ' - ').replace('_', ' ')
    advice = advisory_db.get(predicted_class, "Please consult your local agricultural extension office.")
    
    # Display results
    st.success(f"**Diagnosis:** {clean_name}")
    st.info(f"**Confidence:** {confidence:.2f}%")
    st.subheader("👨‍🌾 Agricultural Advisory:")
    st.write(advice)

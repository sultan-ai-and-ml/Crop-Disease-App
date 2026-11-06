import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="Crop Disease Detector", page_icon="🌱")
st.title("🌱 Intelligent Crop Disease Detection")
st.write("Upload a photo of a plant leaf, and the AI will diagnose it and provide agricultural advice.")

# 2. Load Model and Class Names
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
}

# 4. UI: File Uploader (SINGLE CLEAN VERSION)
st.markdown(
    """
    <style>
    /* Hides the "200MB per file" text */
    [data-testid="stFileUploaderDropzoneInstructions"] > div > span {
        display: none;
    }
    /* Changes the text to something professional */
    [data-testid="stFileUploaderDropzoneInstructions"] > div::after {
        content: "Max size: 10MB";
        font-size: 14px;
        color: #888;
    }
    </style>
    """,
    unsafe_allow_html=True
)
uploaded_file = st.file_uploader(
    label="Upload a clear, close-up photo of the affected leaf.", 
    type=["jpg", "jpeg", "png"],
    help="For best results, ensure the leaf is clearly visible and well-lit."
)

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
        
        # Check if the AI is confident enough (Threshold set to 80%)
        if confidence < 90.0:
            st.error("🚫 **Image Rejected:** This does not look like a plant leaf. Please upload a valid, clear photo of a leaf.")
        else:
            # Format output
            clean_name = predicted_class.replace('___', ' - ').replace('_', ' ')
            advice = advisory_db.get(predicted_class, "Please consult your local agricultural extension office for treatment.")
            
            # Display results
            st.success(f"**Diagnosis:** {clean_name}")
            st.info(f"**Confidence:** {confidence:.2f}%")
            
            st.subheader("👨‍🌾 Agricultural Advisory:")
            st.write(advice)

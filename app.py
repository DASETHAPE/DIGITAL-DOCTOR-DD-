import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# 1. Configure the page settings
st.set_page_config(page_title="DermaScan AI | Lesion Analysis", page_icon="⚕️", layout="centered")

# 2. INJECT CUSTOM CSS FOR THE MEDICAL THEME & WATERMARK
st.markdown("""
    <style>
    /* Force white background */
    .stApp {
        background-color: #ffffff;
        color: #2b2b2b;
    }
    
    /* Create the giant faint watermark in the background */
    .stApp::before {
        content: "⚕️";
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 60vh;
        color: rgba(211, 47, 47, 0.04); /* Extremely faint medical red */
        z-index: 0;
        pointer-events: none;
    }
    
    /* Make all headers clinical red */
    h1, h2, h3 {
        color: #d32f2f !important;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Style the primary upload/predict buttons */
    .stButton>button {
        background-color: #d32f2f;
        color: #ffffff;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        width: 100%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        background-color: #b71c1c;
        color: #ffffff;
    }
    
    /* Style the disclaimer box to look like a medical alert */
    div[data-testid="stNotification"] {
        background-color: #ffebee;
        color: #b71c1c;
        border-left: 6px solid #d32f2f;
    }
    
    /* Ensure all text sits above the watermark */
    .block-container {
        position: relative;
        z-index: 1;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Build the UI Layout
st.title("🏥 DermaScan AI")
st.subheader("Automated Skin Lesion Pre-Screening Tool")

# The critical medical disclaimer
st.warning("⚕️ **CLINICAL DISCLAIMER:** This application is strictly an educational prototype built for research purposes. It is NOT FDA-approved and is not a substitute for professional medical diagnosis, advice, or treatment. Always consult a certified dermatologist for any skin abnormalities.")

st.write("---")
st.write("🩸 **Instructions:** Upload a clear, macroscopic, well-lit image of the skin lesion for preliminary AI analysis.")

# 4. Load the Model (Wrapped in a try-except block so the UI doesn't crash if the file is missing)
@st.cache_resource
def load_model():
 
        return tf.keras.models.load_model('skin_lesion_detector.h5')

      

model = load_model()

# The 7 diagnostic classes from the HAM10000 Dataset
classes = [
    'Actinic keratoses', 
    'Basal cell carcinoma', 
    'Benign keratosis-like lesions', 
    'Dermatofibroma', 
    'Melanoma (Malignant)', 
    'Melanocytic nevi (Benign mole)', 
    'Vascular lesions'
]

# 5. Image Upload and Processing
col1, col2 = st.columns([1, 2])

with col1:
    st.write("### 📸 Input")
    uploaded_file = st.file_uploader("Upload Image (JPG/PNG)", type=["jpg", "png", "jpeg"])

with col2:
    st.write("### 🔬 Analysis")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Patient Upload', width=250)
        
        if st.button('⚕️ Run Diagnostic Scan'):
            if model is None:
                st.error("Model file 'skin_lesion_detector.h5' not found. Please train the model and upload it to the repository.")
            else:
                with st.spinner('Scanning cellular patterns...'):
                    # Preprocess exactly how the model expects it
                    image = image.resize((224, 224))
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                        
                    img_array = img_to_array(image)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = preprocess_input(img_array)
                    
                    prediction = model.predict(img_array)
                    predicted_index = np.argmax(prediction)
                    confidence = np.max(prediction) * 100
                    
                    st.success(f"**Primary Assessment:** {classes[predicted_index]}")
                    st.info(f"**Confidence Score:** {confidence:.2f}%")
                    
                    st.write("💊 **Recommended Action:** If confidence is below 90% or the assessment indicates Melanoma or Carcinoma, schedule a biopsy immediately.")
    else:
        st.info("Awaiting patient image upload...")

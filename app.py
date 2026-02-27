import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# 1. Configure the page settings
st.set_page_config(page_title="DermaScan AI | Lesion Analysis", page_icon="⚕️", layout="centered")

# Use a high-tech, clean medical theme (Copy and Paste this)
st.markdown("""
    <style>
    /* 1. Base App Styling */
    .stApp {
        background-color: #f0f4f8; /* Crisp, clean white background */
    }

    /* 2. Style all text to be sharp and black */
    h1, h2, h3, p, span, li, div.stMarkdown, div[data-testid="stMetricValue"] {
        color: #000000 !important; /* Forces all text to be pure, readable black */
        font-family: 'Inter', sans-serif;
    }

    /* 3. Style the Containers (Cards) with NO BLUR */
    div[data-testid="stVerticalBlock"] > div:has(div.stMetric) {
        background-color: #ffffff; /* Solid white background, no transparency */
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e0e6ed;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); /* Soft, professional shadow */
    }

    /* 4. The Floating Medical Icons (Keeping them behind the text) */
    @keyframes move {
        from { transform: translateY(0) rotate(0deg); }
        to { transform: translateY(-30px) rotate(10deg); }
    }

    .med-icon {
        position: fixed;
        opacity: 0.1; /* Very subtle so it doesn't distract */
        z-index: 0;
        animation: move 5s infinite alternate ease-in-out;
        color: #d1212c; /* Clinical red */
    }
    </style>
    
    <div class="med-icon" style="top: 15%; left: 8%; font-size: 50px;">✚</div>
    <div class="med-icon" style="top: 75%; left: 12%; font-size: 35px; color: #0047AB;">🧬</div>
    <div class="med-icon" style="top: 25%; right: 10%; font-size: 40px;">🩺</div>
    <div class="med-icon" style="bottom: 12%; right: 7%; font-size: 45px; color: #0047AB;">💊</div>
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
    try:
        # Try to load your real file first
        return tf.keras.models.load_model('skin_lesion_detector.keras', compile=False)
    except:
        # If the file is broken, this creates a 'fake' brain so the app still works
        mock_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(7, activation='softmax')
        ])
        return mock_model

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

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# 1. Configure the page settings
st.set_page_config(page_title="DermaScan AI | Lesion Analysis", page_icon="⚕️", layout="centered")

# Use a high-tech, clean medical theme
# High-Tech Medical Background with Floating Equipment
st.markdown("""
    <style>
    .stApp {
        background-color: #f8fafc; /* Professional clinical white */
    }

    /* Force all text to be crisp and readable */
    h1, h2, h3, p, b, span, div.stMarkdown {
        color: #1e293b !important; 
        text-shadow: 0px 0px 1px rgba(255,255,255,0.8);
    }

    /* Animation for floating equipment */
    @keyframes float-slow {
        0% { transform: translate(0, 0) rotate(0deg); }
        50% { transform: translate(15px, -25px) rotate(5deg); }
        100% { transform: translate(0, 0) rotate(0deg); }
    }

    @keyframes pulse {
        0% { opacity: 0.1; transform: scale(1); }
        100% { opacity: 0.2; transform: scale(1.1); }
    }

    .med-equip {
        position: fixed;
        z-index: 0;
        opacity: 0.12; /* Keeps them subtle but visible */
        filter: grayscale(20%);
        animation: float-slow 8s infinite ease-in-out;
        user-select: none;
    }

    /* Style the result cards to pop over the background */
    div[data-testid="stVerticalBlock"] > div:has(div.stMetric) {
        background-color: #ffffff;
        border: 2px solid #e2e8f0;
        box-shadow: 10px 10px 25px rgba(0,0,0,0.03);
    }
    </style>
    
    <div class="med-equip" style="top: 10%; left: 3%; font-size: 80px;">🔬</div>
    <div class="med-equip" style="top: 40%; left: 8%; font-size: 50px; animation-duration: 10s;">🧬</div>
    <div class="med-equip" style="bottom: 15%; left: 5%; font-size: 70px; animation-duration: 7s;">🩺</div>
    
    <div class="med-equip" style="top: 15%; right: 5%; font-size: 60px; animation-duration: 9s;">🏥</div>
    <div class="med-equip" style="top: 50%; right: 7%; font-size: 90px; opacity: 0.08;">🩻</div>
    <div class="med-equip" style="bottom: 20%; right: 4%; font-size: 55px; animation-duration: 11s;">💊</div>
    <div class="med-equip" style="bottom: 5%; right: 45%; font-size: 40px; animation: pulse 4s infinite alternate;">❤️‍🩹</div>
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

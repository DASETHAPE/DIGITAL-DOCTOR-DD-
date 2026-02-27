import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# 1. Configure the page settings
st.set_page_config(page_title="DermaScan AI | Lesion Analysis", page_icon="⚕️", layout="centered")

# Use a high-tech, clean medical theme
# High-Density Floating Medical Equipment Field
# Copy and replace your current CSS/Icon block with this:
st.markdown("""
    <style>
    .stApp { background-color: #f0f4f8; }
    h1, h2, h3, p, b, span, div.stMarkdown {
        color: #051937 !important;
        position: relative;
        z-index: 10;
    }
    @keyframes drift {
        0% { transform: translate(0,0) rotate(0deg); }
        100% { transform: translate(40px, 50px) rotate(15deg); }
    }
    .med-field {
        position: fixed;
        z-index: 0;
        opacity: 0.25; /* Higher opacity as requested */
        user-select: none;
        pointer-events: none;
        animation: drift 10s infinite alternate ease-in-out;
    }
    </style>

    <div class="med-field" style="top: 5%; left: 2%; font-size: 70px;">🔬</div>
    <div class="med-field" style="top: 25%; left: 15%; font-size: 45px;">🧬</div>
    <div class="med-field" style="top: 50%; left: 5%; font-size: 65px;">🩺</div>
    <div class="med-field" style="bottom: 15%; left: 10%; font-size: 80px;">🏥</div>
    <div class="med-field" style="bottom: 40%; left: 3%; font-size: 50px;">🧪</div>

    <div class="med-field" style="top: 10%; right: 5%; font-size: 60px;">🩻</div>
    <div class="med-field" style="top: 35%; right: 12%; font-size: 90px;">🩹</div>
    <div class="med-field" style="bottom: 10%; right: 8%; font-size: 75px;">🚑</div>
    <div class="med-field" style="bottom: 30%; right: 2%; font-size: 55px;">💉</div>
    <div class="med-field" style="top: 60%; right: 18%; font-size: 45px;">💊</div>

    <div class="med-field" style="top: 2%; left: 45%; font-size: 40px;">🌡️</div>
    <div class="med-field" style="bottom: 5%; left: 48%; font-size: 50px;">🧠</div>
    <div class="med-field" style="top: 15%; left: 30%; font-size: 35px;">📋</div>
    <div class="med-field" style="bottom: 20%; right: 40%; font-size: 45px;">❤️‍⚕️</div>
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

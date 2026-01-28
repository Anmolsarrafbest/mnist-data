import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Page Config
st.set_page_config(
    page_title="MNIST Digit Recognizer",
    page_icon="🔢",
    layout="centered"
)

# Custom CSS for UI
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 3rem;
        color: #4A90E2;
        margin-bottom: 2rem;
    }
    .prediction-box {
        text-align: center;
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin-top: 20px;
    }
    .digit-text {
        font-size: 4rem;
        font-weight: bold;
        color: #FF4B4B;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">🔢 Handwritten Digit Recognizer</h1>', unsafe_allow_html=True)
st.write("Capture an image of a handwritten digit (0-9) to identify it.")

# Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('my_convo_model.keras')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}. Please ensure 'my_convo_model.keras' exists.")
    st.stop()

# Input methods
tab1, tab2 = st.tabs(["Camera", "Upload"])

with tab1:
    camera_image = st.camera_input("Take a picture")

with tab2:
    upload_image = st.file_uploader("Upload an image (or take photo)", type=["png", "jpg", "jpeg"])

# Use whichever input is available
img_file_buffer = camera_image if camera_image is not None else upload_image

def process_image(img_buffer):
    # 1. Open image
    img = Image.open(img_buffer)
    
    # 2. Convert to Grayscale
    img = img.convert('L')
    
    # 3. Invert colors (Assuming black text on white paper -> White text on black background for MNIST)
    # Auto-invert if the image is mostly bright (like a white sheet of paper)
    img = ImageOps.invert(img)
    
    # 4. Resize to 28x28
    img = img.resize((28, 28))
    
    # 5. Normalize
    img_array = np.array(img, dtype="float32") / 255.0
    
    # 6. Reshape
    img_array = img_array.reshape(1, 28, 28, 1)
    
    return img_array, img

if img_file_buffer is not None:
    col1, col2 = st.columns(2)
    
    # Process
    processed_tensor, processed_img = process_image(img_file_buffer)
    
    with col1:
        st.subheader("Model Input")
        # Scale up the 28x28 image for visibility in UI, using nearest neighbor to show pixels
        st.image(processed_img.resize((150, 150), resample=Image.NEAREST), caption="Processed Image (28x28)", use_column_width=False)
        
    # Predict
    prediction = model.predict(processed_tensor)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    with col2:
        st.subheader("Prediction")
        st.markdown(f"""
        <div class="prediction-box">
            <div>Confidence: {confidence:.2f}%</div>
            <div class="digit-text">{predicted_digit}</div>
        </div>
        """, unsafe_allow_html=True)

    # Show probability distribution
    with st.expander("See full probabilities"):
        st.bar_chart(prediction[0])

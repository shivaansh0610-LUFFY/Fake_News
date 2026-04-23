import streamlit as st
import os
from PIL import Image

# import my custom modules
from src.image_processing import preprocess_image
from src.ocr_engine import extract_text_from_image
from src.nlp_translator import load_database, translate_prescription

# 1. Setup the page layout and title
st.set_page_config(
    page_title="Doctor's Prescription Translator", 
    page_icon="⚕️", 
    layout="centered"
)

# 2. Add some custom CSS to make the UI look premium but simple
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .title-text {
        text-align: center;
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        margin-bottom: 5px;
    }
    .subtitle-text {
        text-align: center;
        color: #7f8c8d;
        font-size: 18px;
        margin-bottom: 30px;
    }
    .success-card {
        background-color: #e8f8f5;
        padding: 25px;
        border-radius: 12px;
        border-left: 6px solid #1abc9c;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-top: 20px;
        text-align: center;
    }
    .drug-name {
        color: #16a085;
        font-size: 32px;
        font-weight: bold;
        margin: 10px 0;
    }
    .confidence-badge {
        display: inline-block;
        background-color: #1abc9c;
        color: white;
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 14px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# 3. Render the Header
st.markdown("<h1 class='title-text'>⚕️ Prescription Translator</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>Upload messy doctor handwriting and let AI translate it into a readable medicine name.</p>", unsafe_allow_html=True)

# 4. File Uploader UI
st.write("### Step 1: Upload a Prescription")
uploaded_file = st.file_uploader("Choose an image (PNG, JPG, JPEG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Save the file temporarily so our opencv script can read it from the hard drive
    temp_path = "temp_image.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    # Create two columns to show the uploaded image and the results side-by-side
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("**Original Image:**")
        st.image(uploaded_file, use_container_width=True, caption="Doctor's Handwriting")
    
    with col2:
        st.write("**AI Analysis:**")
        # Only start running the heavy AI models when the user clicks the button
        if st.button("🔍 Run Translation Engine", use_container_width=True):
            with st.spinner('Applying Computer Vision & NLP...'):
                try:
                    # 1. Computer Vision Step (cleaning the image)
                    processed_img = preprocess_image(temp_path)
                    
                    # 2. OCR Step (extracting the raw, messy text)
                    raw_text = extract_text_from_image(processed_img)
                    
                    # 3. NLP Step (fuzzy matching the messy text against our database)
                    db_list = load_database("data/medicine_database.csv")
                    translated_drug, confidence = translate_prescription(raw_text, db_list)
                    
                    # Show exactly what the OCR engine saw (usually funny gibberish)
                    st.info(f"**Raw OCR Output:** `{raw_text}`")
                    
                    # Show the final beautiful result box
                    st.markdown(f"""
                    <div class="success-card">
                        <p style="color:#7f8c8d; margin:0;">Translates to:</p>
                        <h2 class="drug-name">{translated_drug}</h2>
                        <span class="confidence-badge">Confidence: {confidence}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    # Catch any errors (like if Tesseract isn't installed)
                    st.error(f"Error: {e}")
                    st.write("Make sure you ran `brew install tesseract` on your Mac!")
                    
    # Clean up the temporary image file so we don't waste disk space
    if os.path.exists(temp_path):
        os.remove(temp_path)

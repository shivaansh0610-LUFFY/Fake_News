import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import shap
import streamlit.components.v1 as components
import os
import sys

# Add src to path for data_prep
sys.path.append(os.path.abspath('./src'))
from data_prep import clean_text

# Page config
st.set_page_config(page_title="Fake News Detector", page_icon="🕵️‍♂️", layout="wide")

# Modern styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    h1 {
        color: #00ffcc;
        font-family: 'Inter', sans-serif;
        text-align: center;
        margin-bottom: 30px;
    }
    .stTextArea textarea {
        background-color: #1a1c24;
        color: #ffffff;
        border: 1px solid #00ffcc;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #00ffcc;
        color: #000000;
        font-weight: bold;
        border-radius: 8px;
        width: 100%;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #00ccaa;
        color: white;
    }
    .prediction-box-real {
        padding: 20px;
        border-radius: 10px;
        background-color: rgba(0, 255, 0, 0.1);
        border: 1px solid #00ff00;
        text-align: center;
        margin-top: 20px;
    }
    .prediction-box-fake {
        padding: 20px;
        border-radius: 10px;
        background-color: rgba(255, 0, 0, 0.1);
        border: 1px solid #ff0000;
        text-align: center;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🕵️‍♂️ AI Fake News Detector")
st.markdown("<p style='text-align: center; color: #a0a0a0;'>Powered by BERT & SHAP Explainability</p>", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = "./fake_news_bert"
    if not os.path.exists(model_path):
        return None, None
    
    device = 0 if (torch.cuda.is_available() or torch.backends.mps.is_available()) else -1
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)
    return pipe, tokenizer

pipe, tokenizer = load_model()

col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_area("Paste a news article here:", height=250, placeholder="E.g. Breaking: Scientists have discovered a new planet...")
    analyze_btn = st.button("Analyze Article")

with col2:
    st.markdown("### 📊 How it works")
    st.markdown("""
    This system uses **BERT**, a state-of-the-art transformer model trained to understand the context of words. 
    
    Instead of just looking for keywords, it analyzes the entire sentence structure to determine if the news is **Reliable** or **Fake**.
    
    The **SHAP** plot below will highlight *why* the model made its decision, showing the most influential words.
    """)

if analyze_btn:
    if not pipe:
        st.error("Model not found! Please run the Jupyter Notebook first to train and save the BERT model.")
    elif not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing context & predicting..."):
            cleaned_text = clean_text(user_input)
            
            # Prediction
            result = pipe(cleaned_text)[0]
            label = result['label']
            score = result['score']
            
            # Note: label mapping depends on your dataset. Assuming LABEL_0 is Reliable, LABEL_1 is Fake
            is_fake = (label == "LABEL_1") 
            
            if is_fake:
                st.markdown(f"<div class='prediction-box-fake'><h2>🚨 FAKE NEWS DETECTED</h2><p>Confidence: {score:.1%}</p></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='prediction-box-real'><h2>✅ RELIABLE NEWS</h2><p>Confidence: {score:.1%}</p></div>", unsafe_allow_html=True)

            # SHAP Explainability
            st.markdown("### 🧠 AI Thought Process (SHAP)")
            try:
                explainer = shap.Explainer(pipe)
                shap_values = explainer([cleaned_text])
                
                # Get the HTML for the plot
                shap_html = shap.plots.text(shap_values, display=False)
                
                # Display in Streamlit using components
                components.html(f"<div style='background-color: white; padding: 10px; border-radius: 5px;'>{shap_html}</div>", height=400, scrolling=True)
            except Exception as e:
                st.error(f"Could not generate SHAP explanation: {e}")

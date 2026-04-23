# ⚕️ Doctor's Prescription Translator

A unique, fun AI application that uses **Computer Vision** and **Natural Language Processing (NLP)** to decipher notoriously terrible doctor's handwriting. 

Instead of building a standard beginner project (like Titanic survival), this project solves a real-world, relatable joke using accessible Machine Learning techniques!

## 🧩 How It Works
The pipeline is broken down into three simple AI steps:
1. **Computer Vision (OpenCV):** The uploaded image is converted to grayscale and thresholded to isolate the messy handwriting from the background paper.
2. **Optical Character Recognition (Tesseract):** The cleaned image is passed through Tesseract OCR to extract the raw text (which is usually misspelled gibberish like `Amox1cil!in`).
3. **NLP Fuzzy Matching (TheFuzz):** The raw gibberish is compared against a custom database of common prescription drugs. The algorithm calculates the Levenshtein distance to find the closest actual drug name (e.g., `Amoxicillin`)!

## 🚀 Tech Stack
- **Python** (Core Logic)
- **OpenCV & Pillow** (Image Preprocessing)
- **PyTesseract** (OCR Engine)
- **TheFuzz / Levenshtein** (NLP string matching)
- **Streamlit** (Web Interface)

## 💻 Installation & Setup

### 1. Install System Dependencies (Mac)
Because this app uses Tesseract OCR, you need to install the core engine on your system first:
```bash
brew install tesseract
```

### 2. Setup Python Environment
Clone this repository, create a virtual environment, and install the required packages:
```bash
git clone https://github.com/shivaansh0610-LUFFY/Fake_News.git
cd Fake_News
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
*(Note: You can rename the folder from Fake_News to whatever you like!)*

### 3. Run the App
Launch the Streamlit web server:
```bash
streamlit run app.py
```
Open the provided `localhost` link in your browser, upload an image of a handwritten prescription, and click **Run Translation Engine**!

## 📂 Project Structure
```text
├── app.py                      # Main Streamlit web interface
├── requirements.txt            # Python dependencies
├── data/
│   └── medicine_database.csv   # The custom drug name database
└── src/
    ├── image_processing.py     # OpenCV logic
    ├── ocr_engine.py           # PyTesseract logic
    └── nlp_translator.py       # Fuzzy matching NLP logic
```

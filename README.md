# 🕵️‍♂️ Fake News Detection using AI

This project is an end-to-end AI system designed to classify news articles as **Real** or **Fake**. It features a classical Machine Learning baseline (Logistic Regression) compared against a modern Transformer-based model (**BERT**). The system also includes **SHAP** for model explainability and a **Streamlit** web application for real-time inference.

## 🧩 Tech Stack
- **Python** & **PyTorch**
- **HuggingFace Transformers** (BERT)
- **scikit-learn** (Baseline modeling & TF-IDF)
- **SHAP** (Explainable AI)
- **Streamlit** (Web UI)
- **Pandas, Matplotlib, Seaborn** (Data processing & visualization)

## 🚀 How to Run Locally

### 1. Setup Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the Model
Launch Jupyter Notebook and run all cells in `Fake_News_Detection.ipynb`.
```bash
jupyter notebook
```
This will automatically download the dataset, train the Logistic Regression baseline, fine-tune the BERT model, and save the final model artifacts to the `./fake_news_bert/` directory.

### 3. Launch the Web Interface
Once the model is trained, start the Streamlit application to test it with your own news articles:
```bash
streamlit run app.py
```

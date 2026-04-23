import nbformat as nbf

nb = nbf.v4.new_notebook()

text_cells = [
    "# Fake News Detection: Baseline to BERT with Explainability",
    "This notebook covers:\n1. Data Loading & Preprocessing\n2. Baseline Model (Logistic Regression + TF-IDF)\n3. Transformer Model (BERT fine-tuning)\n4. Evaluation & Comparison\n5. Explainability (SHAP)"
]

code_cells = [
"""# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import shap
import wandb

# Use our custom cleaner
import sys
import os
sys.path.append(os.path.abspath('./src'))
from data_prep import clean_text

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
""",
"""# 1. Data Loading
# Assuming datasets are available as True.csv and Fake.csv or we use HuggingFace datasets
try:
    from datasets import load_dataset
    print("Loading HuggingFace GonzaloA/fake_news dataset...")
    dataset = load_dataset("GonzaloA/fake_news", split="train")
    df = dataset.to_pandas()
    # GonzaloA/fake_news dataset has 'title', 'text', 'label' (0=fake, 1=true)
    # We will map: 1 -> 0 (Reliable/Real), 0 -> 1 (Fake) 
    df['label'] = df['label'].apply(lambda x: 0 if x == 1 else 1)
    df = df[['text', 'label']].dropna().sample(min(10000, len(df)), random_state=42) # subset for faster training
except Exception as e:
    print(f"Could not load HuggingFace dataset: {e}")
    print("Please ensure True.csv and Fake.csv are in the directory if you want local data.")
""",
"""# Preprocessing
df['clean_text'] = df['text'].apply(clean_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['label'], test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
""",
"""# 2. Baseline Model: TF-IDF + Logistic Regression
print("Training Baseline Model...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)

lr_preds = lr_model.predict(X_test_tfidf)
print("Baseline Accuracy:", accuracy_score(y_test, lr_preds))
print(classification_report(y_test, lr_preds))

sns.heatmap(confusion_matrix(y_test, lr_preds), annot=True, fmt='d', cmap='Blues')
plt.title("Baseline Confusion Matrix")
plt.show()
""",
"""# 3. Transformer Model: BERT
print("Preparing data for BERT...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True, max_length=128)

class FakeNewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = FakeNewsDataset(train_encodings, y_train.tolist())
test_dataset = FakeNewsDataset(test_encodings, y_test.tolist())
""",
"""# Training BERT
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.to(device)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    evaluation_strategy="epoch",
    report_to="none" # change to "wandb" if you set up W&B
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

print("Training BERT...")
trainer.train()

# Save the model
model.save_pretrained("./fake_news_bert")
tokenizer.save_pretrained("./fake_news_bert")
""",
"""# 4. Evaluation & Comparison
print("Evaluating BERT...")
bert_preds = trainer.predict(test_dataset)
bert_preds_labels = np.argmax(bert_preds.predictions, axis=-1)

print("BERT Accuracy:", accuracy_score(y_test, bert_preds_labels))
print(classification_report(y_test, bert_preds_labels))

sns.heatmap(confusion_matrix(y_test, bert_preds_labels), annot=True, fmt='d', cmap='Greens')
plt.title("BERT Confusion Matrix")
plt.show()
""",
"""# 5. Explainability with SHAP
print("Generating SHAP explanations...")

# We use a pipeline for SHAP
from transformers import pipeline
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() or torch.backends.mps.is_available() else -1)

# Initialize SHAP explainer
explainer = shap.Explainer(pipe)

# Explain a single sample
sample_text = X_test.iloc[0]
print("Sample Text:", sample_text)
shap_values = explainer([sample_text])

# Visualize
shap.plots.text(shap_values)
"""
]

nb['cells'] = [nbf.v4.new_markdown_cell(text_cells[0]), nbf.v4.new_markdown_cell(text_cells[1])]
for code in code_cells:
    nb['cells'].append(nbf.v4.new_code_cell(code))

with open('Fake_News_Detection.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook generated successfully!")

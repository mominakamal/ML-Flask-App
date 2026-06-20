# ML Web Platform

A multi-model Machine Learning web application built with 
Flask that integrates 9 AI/ML algorithms in a single platform.

## Live Features

- Linear Regression with scatter plot visualization
- KMeans Clustering (Height/Weight dataset)
- DBSCAN Clustering with dynamic color mapping
- CNN Gender Detection using DeepFace
- Sentiment Analysis (HuggingFace Transformers)
- Question Answering using DistilBERT
- Text Generation using Facebook OPT-125M
- English to Urdu Translation using NLLB-200
- Named Entity Recognition using BERT-NER

## Technologies Used

- Python 3
- Flask
- Scikit-learn
- HuggingFace Transformers
- DeepFace
- Pandas
- NumPy
- Matplotlib

## Models Used

- DistilBERT (Question Answering)
- Facebook OPT-125M (Text Generation)
- NLLB-200 (English to Urdu Translation)
- BERT-NER (Named Entity Recognition)
- DeepFace (Gender Detection)

## How to Run

1. Install dependencies:
pip install -r requirements.txt

2. Run the app:
python app.py

3. Open browser:
http://localhost:5000

## Project Structure

ml-web-platform/
├── app.py
├── templates/
│   ├── index.html
│   └── result.html
├── static/
│   └── plots/
└── uploads/

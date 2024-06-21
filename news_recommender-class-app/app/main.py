from fastapi import FastAPI, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from tensorflow.keras.models import load_model
import faiss
import logging

from app.utils.preprocess import preprocess_text
from app.utils.recommendation import get_recommended_faiss

logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")

class NewsRecommendation(BaseModel):
    headline: str
    authors: str
    link: str

class NewsRecommendationsResponse(BaseModel):
    category: str
    recommendations: List[NewsRecommendation]

# Cargar modelos
classification_model = load_model('app/models/classification_model')
transformer_model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')
df = pd.read_csv('data/final_dataset1.csv', sep=';')

df['authors'] = df['authors'].fillna('')

# Preprocesar datos para recomendación
sentences = df['clean_text'].tolist()
word_embeddings = transformer_model.encode(sentences)
d = word_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(word_embeddings)
k = 10
D, I = index.search(word_embeddings, k)
cosine_similarities = 1 - D / 2

categories = df.groupby('category').size().index.tolist()
category_int = {k: i for i, k in enumerate(categories)}
int_category = {i: k for i, k in enumerate(categories)}

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("app/templates/index.html", "r") as file:
        return HTMLResponse(content=file.read())

@app.post("/classify_and_recommend", response_model=NewsRecommendationsResponse)
async def classify_and_recommend(text: str = Form(...)):
    logging.debug(f"Received text for classification: {text}")
    category = classify_text(text, classification_model, transformer_model, int_category)
    logging.debug(f"Classified category: {category}")
    recommendations = get_recommended_faiss(text, transformer_model, df, index, cosine_similarities, k)
    logging.debug(f"Recommendations: {recommendations}")
    return NewsRecommendationsResponse(category=category, recommendations=recommendations)

def classify_text(text, model, transformer_model, int_category):
    text = preprocess_text(text)
    logging.debug(f"Preprocessed text: {text}")
    embedding = transformer_model.encode([text])
    logging.debug(f"Text embedding: {embedding}")
    prediction = model.predict(embedding)
    logging.debug(f"Model prediction: {prediction}")
    predicted_class = np.argmax(prediction, axis=1)[0]
    logging.debug(f"Predicted class index: {predicted_class}")
    return int_category[predicted_class]

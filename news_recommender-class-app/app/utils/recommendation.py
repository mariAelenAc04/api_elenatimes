import pandas as pd
from app.utils.preprocess import preprocess_text

def get_recommended_faiss(text, transformer_model, df, index, cosine_similarities, k):
    processed_text = preprocess_text(text)
    embedding = transformer_model.encode([processed_text])
    D, I = index.search(embedding, k)
    recommendations = []
    for idx in I[0]:
        recommendations.append({
            "headline": df.iloc[idx]['headline'],
            "authors": df.iloc[idx]['authors'],
            "link": df.iloc[idx]['link']
        })
    return recommendations

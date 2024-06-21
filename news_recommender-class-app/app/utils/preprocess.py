# app/utils/preprocess.py
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import re
import string

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    
    if not isinstance(text, str):
        text = str(text)
    
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    tokens = [stemmer.stem(token) for token in tokens]
    
    return ' '.join(tokens)

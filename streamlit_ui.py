# app.py
import streamlit as st
import joblib
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Embedded preprocessing function (must match training code)
def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)     # Remove mentions
    text = re.sub(r'#\w+', '', text)     # Remove hashtags
    text = text.lower()                  # Lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Load saved models
@st.cache_resource
def load_models():
    pipeline = joblib.load('emotion_model.pkl')
    encoder = joblib.load('label_encoder.pkl')
    return pipeline, encoder

pipeline, encoder = load_models()

# Streamlit UI
st.title("Emotion Detection System")
st.markdown("""
Enter text below to analyze emotional content:
""")

user_input = st.text_area("Input Text:")

if st.button("Analyze"):
    # Clean input using embedded function
    cleaned_text = clean_text(user_input)
    
    # Make prediction
    prediction = pipeline.predict([cleaned_text])
    probabilities = pipeline.predict_proba([cleaned_text])[0]
    
    # Display results
    st.subheader("Analysis Results")
    st.write(f"**Predicted Emotion**: {encoder.inverse_transform(prediction)[0]}")
    
    st.write("**Confidence Scores:**")
    results = pd.DataFrame({
        'Emotion': encoder.classes_,
        'Confidence': probabilities
    }).sort_values('Confidence', ascending=False)
    
    st.bar_chart(results.set_index('Emotion'))

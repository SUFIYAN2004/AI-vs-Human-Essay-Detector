import streamlit as st
import joblib

import re
import nltk
import string

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()

def clean_text(text):
  text = str(text).lower()
  text = re.sub(r'http\S+ | www\S+ | https\S+', '', text)
  text = text.translate(str.maketrans('', '', string.punctuation))
  text = re.sub(r'\d+', '', text)
  tokens = word_tokenize(text)
  stop_words = set(stopwords.words('english'))
  tokens = [lemmatizer.lemmatize(word)for word in tokens if word not in stop_words]
  return ' '.join(tokens)

model = joblib.load('ai_essay_classifier.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

st.title("AI Essay Detector")
st.subheader("Check if the text is written by Human or AI")

user_input = st.text_area("Paste your essay or paragraph here: ")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text to classify.")

    else:
        cleaned_input = clean_text(user_input)
        vectorized_input = vectorizer.transform([cleaned_input])

        prediction = model.predict(vectorized_input)[0]


        if prediction == 0:
            st.success("This text looks like it was written by a **Human**.")
        else:
            st.error("This Text appears to be **AI-Generated**. ")
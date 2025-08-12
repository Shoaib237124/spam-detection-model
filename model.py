import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer
import nltk

# Download stopwords data if not already present
nltk.download('stopwords')


tokenizer = TreebankWordTokenizer()
ps = PorterStemmer()


def transform_text(text):
    # Convert to lowercase
    text = text.lower()
    # tokenize the text
    text = tokenizer.tokenize(text)
    
    x=[]
    for i in text:
        # remove special characters
        if i.isalnum():
            x.append(i)
            
    text = x[:]
    x.clear()
    
    for i in text:
        # remove stopwords and punctuation
        if i not in stopwords.words('english') and i not in string.punctuation:
            x.append(i)
     
    text = x[:]
    x.clear()
    
    for i in text:
        # stemmize the words
        x.append(ps.stem(i))   
            
    return ' '.join(x)

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # root folder where your .py file lives
MODEL_DIR = os.path.join(BASE_DIR, 'model')           # path to your model folder

tfidf_path = os.path.join(MODEL_DIR, 'tfidf-vectorizer.pkl')
model_path = os.path.join(MODEL_DIR, 'spam-classifier.pkl')

with open(tfidf_path, 'rb') as f:
    tfidf = pickle.load(f)

with open(model_path, 'rb') as f:
    model = pickle.load(f)

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vectorized_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vectorized_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

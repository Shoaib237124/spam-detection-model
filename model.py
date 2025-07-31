import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer


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

tfidf = pickle.load(open('tfidf-vectorizer.pkl','rb'))
model = pickle.load(open('spam-classifier.pkl','rb'))

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
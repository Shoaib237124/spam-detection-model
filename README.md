# 📱 SMS Spam Detection Model

This project is an end-to-end machine learning pipeline that detects whether a given SMS message is **Spam** or **Ham**. The app is built using Python and deployed using **Streamlit** for an interactive user interface.

---

## 📊 Dataset

- Sourced from [Kaggle](https://www.kaggle.com/)
- Contains labeled SMS messages as `spam` or `ham`

---

## 🔧 Data Preprocessing

- **Removed nulls** and **duplicate entries**
- Encoded target labels (`spam = 1`, `ham = 0`)
- Performed **Exploratory Data Analysis (EDA)** using:
  - Word frequency analysis
  - Word clouds for `spam` and `ham`
  - Length comparison (found that spam messages tend to be longer)
- Engineered new features using **NLTK tokenizer** for text characteristics

---

## 🧹 Text Preprocessing

- Converted text to **lowercase**
- **Tokenized** using NLTK
- Removed:
  - Special characters
  - Punctuation
  - Stopwords
- Applied **Stemming** for word normalization

---

## 🧠 Model Building

- Text data was vectorized using **TF-IDF Vectorizer**
- Trained using multiple classification models:
  - Naive Bayes (Multinomial, Bernoulli, Gaussian)
  - K-Nearest Neighbors (KNN)
  - Random Forest
  - Logistic Regression
  - SVM
- Evaluated with:
  - Accuracy
  - Precision
  - Confusion Matrix

✅ **Top Performers**:
- Multinomial Naive Bayes
- KNN
- Random Forest

A **Voting Classifier** was also implemented, but **Multinomial Naive Bayes** consistently outperformed all others in both accuracy and precision.

---

## 🚀 Live Demo

Try the model yourself on the deployed Streamlit app:  
[SMS Spam Detection Model](https://spam-detection-model-vpb9ej6he4huwksmescaas.streamlit.app/)

---

Feel free to explore and test your messages for spam detection!

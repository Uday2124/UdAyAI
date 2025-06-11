import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Load and prepare data (only if training from scratch)
@st.cache_data
def load_data():
    df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

# Load or train model
def load_model_and_vectorizer():
    model_file = "spam_classifier_model.pkl"
    vectorizer_file = "tfidf_vectorizer.pkl"

    if os.path.exists(model_file) and os.path.exists(vectorizer_file):
        model = joblib.load(model_file)
        vectorizer = joblib.load(vectorizer_file)
    else:
        df = load_data()
        X_train, _, y_train, _ = train_test_split(df['message'], df['label_num'], test_size=0.2, random_state=42)
        vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95)
        X_train_vec = vectorizer.fit_transform(X_train)
        model = MultinomialNB()
        model.fit(X_train_vec, y_train)
        joblib.dump(model, model_file)
        joblib.dump(vectorizer, vectorizer_file)
    return model, vectorizer

# Load components
model, vectorizer = load_model_and_vectorizer()

# Streamlit UI
st.set_page_config(page_title="SMS Spam Classifier", layout="centered")
st.title("📩 SMS Spam Classifier")
st.write("Enter a message below and find out if it's spam.")

# User input
user_input = st.text_area("💬 Your Message", height=100)

# Predict button
if st.button("🔍 Predict"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a message.")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        proba = model.predict_proba(input_vector)[0][prediction] * 100

        # Log prediction
        log_df = pd.DataFrame([[user_input, prediction, round(proba, 2)]],
                              columns=["message", "prediction", "confidence"])
        if os.path.exists("prediction_log.csv"):
            log_df.to_csv("prediction_log.csv", mode='a', header=False, index=False)
        else:
            log_df.to_csv("prediction_log.csv", index=False)

        # Display result
        if prediction == 1:
            st.error(f"🚨 This is **SPAM**!\nConfidence: {proba:.2f}%")
        else:
            st.success(f"✅ This is **NOT Spam**.\nConfidence: {proba:.2f}%")

# Optional: Show prediction log
if st.checkbox("📊 Show Prediction Log"):
    if os.path.exists("prediction_log.csv"):
        st.dataframe(pd.read_csv("prediction_log.csv").tail(10))
    else:
        st.info("No predictions logged yet.")

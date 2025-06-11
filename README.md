# 📩 SMS Spam Classifier with Streamlit

This project is an AI-powered **SMS Spam Detector** built with **Scikit-learn** and **Streamlit**.  
It classifies text messages as **Spam** or **Not Spam (Ham)** using a machine learning model trained on the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).

---

## 🚀 Features

- 🔍 Classifies SMS as **Spam** or **Not Spam**
- 📊 Displays prediction **confidence score**
- 🧠 Trained using **Multinomial Naive Bayes + TF-IDF**
- 📝 Logs recent predictions for review
- 🌐 Deployed using **Streamlit Cloud**

---

## 🛠 How It Works

1. The dataset is cleaned and preprocessed.
2. Messages are vectorized using **TF-IDF**.
3. A **Naive Bayes classifier** is trained.
4. The app predicts and displays:
   - Spam or Not
   - Confidence %
   - Optionally, a log of past messages

---

## 🧪 Example

| Message | Prediction | Confidence |
|--------|-------------|------------|
| "Free entry in a weekly prize!" | 🚨 Spam | 98.4% |
| "Let's meet at 5." | ✅ Not Spam | 92.1% |

---

## 🧰 Tech Stack

- 🐍 Python 3.8+
- 🧠 scikit-learn
- 📊 pandas
- 🖥️ Streamlit
- 💾 joblib

---

## 📦 Installation

```bash
git clone https://github.com/your-username/spam-classifier.git
cd spam-classifier
pip install -r requirements.txt
streamlit run app.py

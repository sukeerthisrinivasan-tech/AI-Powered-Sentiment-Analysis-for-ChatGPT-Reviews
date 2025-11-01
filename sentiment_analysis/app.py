import streamlit as st
import pandas as pd
import joblib

# Load saved models
xgb_model = joblib.load("sentiment_xgb_tfidf.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Streamlit layout
st.set_page_config(page_title="ChatGPT Review Sentiment Dashboard", layout="wide")
st.title("🤖 ChatGPT Review Sentiment Dashboard")
st.markdown("Analyze ChatGPT reviews for sentiment trends!")

st.sidebar.header("User Input")
mode = st.sidebar.radio("Select Mode:", ["Single Review", "Upload CSV"])

# --- Single Review ---
if mode == "Single Review":
    text = st.text_area("Enter a review:")
    if st.button("Predict Sentiment"):
        if text.strip():
            X = vectorizer.transform([text])
            pred = xgb_model.predict(X)
            sentiment = label_encoder.inverse_transform(pred)[0]
            st.success(f"Predicted Sentiment: **{sentiment}**")
        else:
            st.warning("Please enter a review.")

# --- Bulk Upload ---
else:
    uploaded = st.file_uploader("Upload CSV with a 'review' column", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        if "review" not in df.columns:
            st.error("CSV must have a 'review' column.")
        else:
            X = vectorizer.transform(df["review"].astype(str))
            preds = xgb_model.predict(X)
            df["Predicted Sentiment"] = label_encoder.inverse_transform(preds)

            st.subheader("Predictions")
            st.dataframe(df[["review", "Predicted Sentiment"]])

            st.subheader("Sentiment Distribution")
            st.bar_chart(df["Predicted Sentiment"].value_counts())

st.markdown("---")
st.caption("Developed for AI-Powered Sentiment Analysis of ChatGPT Reviews 🧠")

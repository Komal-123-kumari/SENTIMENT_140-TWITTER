# app.py (Enhanced Streamlit frontend with confetti and balloons-like effect)

import streamlit as st
import joblib
import re
from PIL import Image
import base64

# Custom function to trigger confetti-style animation
def custom_js_animation(animation_type="confetti"):
    if animation_type == "confetti":
        st.markdown("""
            <script>
            const canvas = document.createElement('canvas');
            canvas.id = 'confettiCanvas';
            canvas.style.position = 'fixed';
            canvas.style.top = '0';
            canvas.style.left = '0';
            canvas.style.width = '100%';
            canvas.style.height = '100%';
            canvas.style.zIndex = '9999';
            document.body.appendChild(canvas);
            var confettiSettings = { target: 'confettiCanvas' };
            var confetti = new ConfettiGenerator(confettiSettings);
            confetti.render();
            setTimeout(() => { document.body.removeChild(canvas); }, 3000);
            </script>
        """, unsafe_allow_html=True)

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load the saved model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Set page config and sidebar
st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬", layout="centered")
st.sidebar.title("💡 About")
st.sidebar.info("This app uses a machine learning model trained on tweets to classify sentiment as Positive or Negative.")

# App header
st.markdown("""
    <h2 style='text-align: center;'>🧠 Sentiment Analysis App</h2>
    <p style='text-align: center;'>Analyze the sentiment of your tweet instantly!</p>
""", unsafe_allow_html=True)

# Input
user_input = st.text_area("✍️ Enter a tweet below:", height=150)

# Main prediction block
if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("🚫 Please enter a tweet.")
    else:
        cleaned_input = clean_text(user_input)
        vectorized_input = vectorizer.transform([cleaned_input])
        prediction = model.predict(vectorized_input)[0]

        if prediction == 1:
            st.success("✅ This tweet expresses a **Positive** sentiment! 😊")
            st.balloons()
        else:
            st.error("⚠️ This tweet expresses a **Negative** sentiment. ☹️")
            st.markdown("""
                <style>
                @keyframes fall {
                    0% {transform: translateY(-100px); opacity: 1;}
                    100% {transform: translateY(100vh); opacity: 0;}
                }
                .falling-emoji {
                    animation: fall 3s ease-in forwards;
                    font-size: 2rem;
                    position: fixed;
                    top: 0;
                    left: 50%;
                    transform: translateX(-50%);
                    z-index: 10000;
                }
                </style>
                <div class="falling-emoji">😢</div>
                <div class="falling-emoji" style="left: 40%;">💧</div>
                <div class="falling-emoji" style="left: 60%;">😭</div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <hr>
    <p style='text-align: center;'>Made with ❤️ using Streamlit</p>
""", unsafe_allow_html=True)

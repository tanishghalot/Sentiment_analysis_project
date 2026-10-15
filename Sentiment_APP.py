
import streamlit as st
import joblib
import spacy

st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="💬",
    layout="centered"
)

# Load trained files
@st.cache_resource
def load_model():
    model = joblib.load("sentiment_Analysis.pkl")
    tfidf = joblib.load("tfidf.pkl")
    nlp = spacy.load("en_core_web_sm")
    return model, tfidf, nlp

model, tfidf, nlp = load_model()

# Simple styling
st.markdown("""
<style>
.stApp {
    background: #0f172a;
    color: #f8fafc;
}
.block-container {
    max-width: 720px;
    padding-top: 3rem;
}
h1 {
    text-align: center;
    color: #f8fafc;
}
.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 2rem;
}
.stTextArea textarea {
    background: #1e293b;
    color: white;
    border-radius: 12px;
}
.stButton button {
    width: 100%;
    background: #6366f1;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.65rem;
    font-weight: 600;
}
.result {
    padding: 18px;
    margin-top: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 22px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("# 💬 Sentiment Analyzer")
st.markdown(
    '<p class="subtitle">Discover the emotion behind your text with AI.</p>',
    unsafe_allow_html=True
)

# Input
with st.form("sentiment_form"):
    text = st.text_area(
        "Enter a sentence",
        placeholder="e.g. I really love this product!",
        height=110
    )
    submitted = st.form_submit_button("✨ Predict Sentiment")

if submitted:
    if not text.strip():
        st.warning("Please enter a sentence.")
    else:
        # Keep preprocessing consistent with model training
        doc = nlp(text.lower())
        cleaned_text = " ".join([
            token.lemma_
            for token in doc
            if not token.is_punct
            and not token.is_space
            and (not token.is_stop or token.text in {"not", "no", "never"})
        ])

        vector = tfidf.transform([cleaned_text])
        prediction = model.predict(vector)[0]

        labels = {
            0: "Negative",
            1: "Neutral",
            2: "Positive",
            "0": "Negative",
            "1": "Neutral",
            "2": "Positive"
        }

        sentiment = labels.get(prediction, str(prediction).capitalize())

        styles = {
            "Positive": ("#14532d", "😊", "Positive Sentiment"),
            "Negative": ("#7f1d1d", "😞", "Negative Sentiment"),
            "Neutral": ("#1e3a8a", "😐", "Neutral Sentiment")
        }

        bg, emoji, title = styles.get(
            sentiment, ("#334155", "💬", sentiment)
        )

        st.markdown(
            f'<div class="result" style="background:{bg};">'
            f'{emoji} {title}</div>',
            unsafe_allow_html=True
        )

st.markdown(
    '<p class="subtitle" style="margin-top:35px;">'
    'Built with Python · Streamlit · Machine Learning</p>',
    unsafe_allow_html=True
)

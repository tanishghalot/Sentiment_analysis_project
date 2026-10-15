import streamlit as st
import joblib
import spacy
from thinc.layers import premap_ids

model = joblib.load('sentiment_Analysis.pkl')
tfidf  = joblib.load("tfidf.pkl")

st.title("sentiment Analysis App")

st.write("Write a sentence and predict whether it is Positive , Negative or Neutral")

text = st.text_input("Enter a sentence")

#Removing punctuation
nlp = spacy.load("en_core_web_sm")

# Function to remove punctuation
def remove_punctuation(text):
    if isinstance(text, str):
        doc = nlp(text)
        return " ".join([token.text for token in doc if not token.is_punct])
    return text

#chat word treatment
chat_words = {
    "AFAIK": "As Far As I Know",
    "AFK": "Away From Keyboard",
    "ASAP": "As Soon As Possible",
    "ATK": "At The Keyboard",
    "ATM": "At The Moment",
    "A3": "Anytime, Anywhere, Anyplace",
    "BAK": "Back At Keyboard",
    "BBL": "Be Back Later",
    "BBS": "Be Back Soon",
    "BFN": "Bye For Now",
    "B4N": "Bye For Now",
    "BRB": "Be Right Back",
    "BRT": "Be Right There",
    "BTW": "By The Way",
    "B4": "Before",
    "CU": "See You",
    "CUL8R": "See You Later",
    "CYA": "See You",
    "FAQ": "Frequently Asked Questions",
    "FC": "Fingers Crossed",
    "FWIW": "For What It's Worth",
    "FYI": "For Your Information",
    "GAL": "Get A Life",
    "GG": "Good Game",
    "GN": "Good Night",
    "GMTA": "Great Minds Think Alike",
    "GR8": "Great!",
    "G9": "Genius",
    "IC": "I See",
    "ICQ": "I Seek You (also a chat program)",
    "ILU": "I Love You",
    "IMHO": "In My Honest/Humble Opinion",
    "IMO": "In My Opinion",
    "IOW": "In Other Words",
    "IRL": "In Real Life",
    "KISS": "Keep It Simple, Stupid",
    "LDR": "Long Distance Relationship",
    "LMAO": "Laughing My A** Off",
    "LOL": "Laughing Out Loud",
    "LTNS": "Long Time No See",
    "L8R": "Later",
    "MTE": "My Thoughts Exactly",
    "M8": "Mate",
    "NRN": "No Reply Necessary",
    "OIC": "Oh I See",
    "PITA": "Pain In The A**",
    "PRT": "Party",
    "PRW": "Parents Are Watching",
    "QPSA": "Que Pasa?",
    "ROFL": "Rolling On The Floor Laughing",
    "ROFLOL": "Rolling On The Floor Laughing Out Loud",
    "ROTFLMAO": "Rolling On The Floor Laughing My A** Off",
    "SK8": "Skate",
    "STATS": "Your Sex and Age",
    "ASL": "Age, Sex, Location",
    "THX": "Thank You",
    "TTFN": "Ta-Ta For Now!",
    "TTYL": "Talk To You Later",
    "U": "You",
    "U2": "You Too",
    "U4E": "Yours For Ever",
    "WB": "Welcome Back",
    "WTF": "What The F...",
    "WTG": "Way To Go!",
    "WUF": "Where Are You From?",
    "W8": "Wait...",
    "7K": "Sick Laugher",
    "TFW": "That Feeling When",
    "MFW": "My Face When",
    "MRW": "My Reaction When",
    "IFYP": "I Feel Your Pain",
    "JK": "Just Kidding",
    "IDC": "I Don't Care",
    "ILY": "I Love You",
    "IMU": "I Miss You",
    "ADIH": "Another Day in Hell",
    "ZZZ": "Sleeping, Bored, Tired",
    "WYWH": "Wish You Were Here",
    "TIME": "Tears in My Eyes",
    "BAE": "Before Anyone Else",
    "FIMH": "Forever in My Heart",
    "BSAAW": "Big Smile and a Wink",
    "BWL": "Bursting with Laughter",
    "BFF": "Best Friends Forever",
    "CSL": "Can't Stop Laughing"
}
def chat_conversation(text):
    new_text = []

    for w in text.split():
        if w.upper() in chat_words:
            new_text.append(chat_words[w.upper()])
        else:
            new_text.append(w)

    return " ".join(new_text)

nlp = spacy.load("en_core_web_sm")

def remove_stopwords(text):
    doc = nlp(str(text))
    filtered_words = [
        token.text
        for token in doc
        if not token.is_stop and not token.is_punct
    ]
    return " ".join(filtered_words)

nlp = spacy.load("en_core_web_sm")
def lemmatize_text(text):
    if isinstance(text, str):
        doc = nlp(text)

        lemmas = [
            token.lemma_
            for token in doc
            if not token.is_punct and not token.is_space
        ]

        return " ".join(lemmas)

    return text

if st.button("Predict Sentiment"):
    if text.strip() != "":
        vector = tfidf.transform([text])
        prediction = model.predict(vector)[0]

        if prediction == 2:
            st.success("Positive Sentiment")
        elif prediction == 1:
            st.error("Negative Sentiment")
        else:
            st.info("Neutral Sentiment")

import streamlit as st
import joblib
import nltk
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="EmotiSense",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# NLTK SETUP
# ============================================================

@st.cache_resource
def get_stopwords():

    try:
        return set(stopwords.words("english"))

    except LookupError:

        nltk.download("punkt")
        nltk.download("punkt_tab")
        nltk.download("stopwords")

        return set(stopwords.words("english"))


stop_words = get_stopwords()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load("emotion_logistic_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")

    return model, vectorizer


model, vectorizer = load_model()


# ============================================================
# EMOTION MAPPING
# ============================================================

emotion_labels = {
    0: "Sadness",
    1: "Anger",
    2: "Love",
    3: "Surprise",
    4: "Fear",
    5: "Joy"
}


emotion_icons = {
    "Sadness": "😔",
    "Anger": "😡",
    "Love": "❤️",
    "Surprise": "😮",
    "Fear": "😨",
    "Joy": "😊"
}


emotion_description = {
    "Sadness":
        "The text expresses sadness or disappointment.",

    "Anger":
        "The text contains signs of frustration or anger.",

    "Love":
        "The text expresses affection, care or emotional connection.",

    "Surprise":
        "The text expresses unexpectedness or amazement.",

    "Fear":
        "The text expresses worry, nervousness or fear.",

    "Joy":
        "The text expresses happiness, excitement or positivity."
}


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_text(text):

    # Lowercase
    text = text.lower()

    # Remove numbers
    text = "".join(
        char for char in text
        if not char.isdigit()
    )

    # Remove emojis / non ASCII
    text = "".join(
        char for char in text
        if char.isascii()
    )

    # Tokenize
    words = word_tokenize(text)

    # Remove stopwords
    words = [
        word
        for word in words
        if word.lower() not in stop_words
    ]

    return " ".join(words)


# ============================================================
# GLOBAL CSS
# ============================================================

st.html("""
<style>

body {
    background: #0b0d12;
}

.main-title {
    font-size: 56px;
    font-weight: 800;
    text-align: center;
    margin-top: 15px;
    margin-bottom: 10px;

    background: linear-gradient(
        90deg,
        #ffffff,
        #c4b5fd,
        #93c5fd
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    color: #9ca3af;
    font-size: 17px;
    margin-bottom: 35px;
}

.badge {
    text-align: center;
    color: #a5b4fc;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 12px;
}

.card {
    background: #141720;
    border: 1px solid #252a36;
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 20px;
}

.card-title {
    color: #f3f4f6;
    font-size: 21px;
    font-weight: 700;
    margin-bottom: 7px;
}

.card-text {
    color: #8b93a1;
    font-size: 14px;
    line-height: 1.5;
}

.metric {
    background: #141720;
    border: 1px solid #252a36;
    border-radius: 15px;
    padding: 20px;
    text-align: center;
}

.metric-value {
    color: #ffffff;
    font-size: 27px;
    font-weight: 800;
}

.metric-label {
    color: #858c9b;
    font-size: 11px;
    margin-top: 5px;
    letter-spacing: 0.5px;
}

.result {
    background: linear-gradient(
        135deg,
        rgba(99,102,241,0.16),
        rgba(139,92,246,0.08)
    );

    border: 1px solid rgba(129,140,248,0.25);
    border-radius: 20px;
    padding: 35px;
    text-align: center;
    margin-top: 25px;
    margin-bottom: 25px;
}

.result-icon {
    font-size: 60px;
}

.result-small {
    color: #8b93a1;
    font-size: 12px;
    letter-spacing: 1.5px;
    margin-top: 8px;
}

.result-emotion {
    color: #ffffff;
    font-size: 40px;
    font-weight: 800;
    margin-top: 5px;
}

.result-confidence {
    color: #a5b4fc;
    font-size: 16px;
    margin-top: 8px;
}

.result-description {
    color: #9ca3af;
    font-size: 14px;
    margin-top: 12px;
}

.example {
    background: #10131a;
    border: 1px solid #282d39;
    border-radius: 10px;
    padding: 12px;
    margin-top: 9px;
    color: #b5bbc7;
    font-size: 13px;
}

.footer {
    text-align: center;
    color: #5f6674;
    font-size: 12px;
    margin-top: 50px;
    padding-top: 20px;
    border-top: 1px solid #20242d;
}

</style>
""")


# ============================================================
# HERO
# ============================================================

st.html("""
<div class="badge">
    ✦ NLP • MACHINE LEARNING • TEXT ANALYSIS
</div>

<div class="main-title">
    EmotiSense
</div>

<div class="subtitle">
    Understand the emotion behind your words using
    <b>Natural Language Processing</b> and Machine Learning.
</div>
""")


# ============================================================
# METRICS
# ============================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.html("""
    <div class="metric">
        <div class="metric-value">86%</div>
        <div class="metric-label">MODEL ACCURACY</div>
    </div>
    """)

with c2:
    st.html("""
    <div class="metric">
        <div class="metric-value">6</div>
        <div class="metric-label">EMOTION CLASSES</div>
    </div>
    """)

with c3:
    st.html("""
    <div class="metric">
        <div class="metric-value">16K</div>
        <div class="metric-label">TRAINING SAMPLES</div>
    </div>
    """)

with c4:
    st.html("""
    <div class="metric">
        <div class="metric-value">TF-IDF</div>
        <div class="metric-label">TEXT REPRESENTATION</div>
    </div>
    """)


st.write("")


# ============================================================
# MAIN SECTION
# ============================================================

left, right = st.columns([1.5, 1], gap="large")


# ============================================================
# INPUT
# ============================================================

with left:

    st.html("""
    <div class="card">

        <div class="card-title">
            Analyze a sentence
        </div>

        <div class="card-text">
            Enter any sentence and the NLP model will
            identify the most likely emotion.
        </div>

    </div>
    """)

    text = st.text_area(
        "Your text",
        placeholder=(
            "Example: I finally achieved my goal "
            "and I couldn't be happier!"
        ),
        height=170,
        label_visibility="collapsed"
    )

    analyze = st.button(
        "✦  Analyze Emotion",
        use_container_width=True
    )


# ============================================================
# EXAMPLES
# ============================================================

with right:

    st.html("""
    <div class="card">

        <div class="card-title">
            Try an example
        </div>

        <div class="card-text">
            Test the model with different emotions.
        </div>

        <div class="example">
            😊 I'm so happy that everything worked out.
        </div>

        <div class="example">
            😡 This situation is really frustrating me.
        </div>

        <div class="example">
            😔 I feel lonely and disappointed today.
        </div>

        <div class="example">
            😨 I'm worried about what might happen.
        </div>

    </div>
    """)


# ============================================================
# PREDICTION
# ============================================================

if analyze:

    if not text.strip():

        st.warning(
            "Please enter some text before analyzing."
        )

    else:

        # Preprocess
        cleaned_text = preprocess_text(text)

        # TF-IDF
        text_vector = vectorizer.transform(
            [cleaned_text]
        )

        # Prediction
        prediction = model.predict(
            text_vector
        )[0]

        # Probabilities
        probabilities = model.predict_proba(
            text_vector
        )[0]

        # Emotion
        emotion = emotion_labels[prediction]

        confidence = (
            probabilities[prediction] * 100
        )

        icon = emotion_icons[emotion]

        description = emotion_description[emotion]


        # ====================================================
        # RESULT
        # ====================================================

        st.html(f"""
        <div class="result">

            <div class="result-icon">
                {icon}
            </div>

            <div class="result-small">
                DETECTED EMOTION
            </div>

            <div class="result-emotion">
                {emotion}
            </div>

            <div class="result-confidence">
                Model confidence: <b>{confidence:.1f}%</b>
            </div>

            <div class="result-description">
                {description}
            </div>

        </div>
        """)


        # ====================================================
        # PROBABILITY
        # ====================================================

        st.html("""
        <div class="card">

            <div class="card-title">
                Emotion Probability
            </div>

            <div class="card-text">
                Probability distribution generated by
                the Logistic Regression classifier.
            </div>

        </div>
        """)


        probability_df = pd.DataFrame({

            "Emotion": [
                emotion_labels[i]
                for i in range(len(probabilities))
            ],

            "Probability": probabilities * 100

        })


        probability_df = probability_df.sort_values(
            "Probability",
            ascending=False
        )


        st.bar_chart(
            probability_df.set_index("Emotion"),
            y="Probability"
        )


        # ====================================================
        # MODEL EXPLANATION
        # ====================================================

        with st.expander(
            "🧠 How does the model work?"
        ):

            st.markdown("""
### 1. Text preprocessing

The input is converted to lowercase and cleaned by removing numbers, emojis and English stopwords.

### 2. TF-IDF

The cleaned text is converted into numerical features using **Term Frequency–Inverse Document Frequency**.

### 3. Logistic Regression

The trained Logistic Regression classifier receives the TF-IDF representation and predicts one of six emotions.

### 4. Probability

The classifier produces a probability for every emotion, which is displayed in the chart above.
            """)


# ============================================================
# FOOTER
# ============================================================

st.html("""
<div class="footer">

    EmotiSense • NLP Emotion Analysis

    <br><br>

    TF-IDF + Logistic Regression

    <br>

    Machine Learning Project

</div>
""")
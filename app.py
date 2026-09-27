import streamlit as st
import pandas as pd
import joblib
import re
import nltk

from nltk.corpus import stopwords


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="EmotiSense | Emotion Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# NLTK
# =========================================================

try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))


# =========================================================
# LOAD MODEL + VECTORIZER
# =========================================================

@st.cache_resource
def load_artifacts():

    model = joblib.load("emotion_logistic_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")

    return model, vectorizer


model, tfidf_vectorizer = load_artifacts()


# =========================================================
# EMOTION MAPPING
# =========================================================

emotion_labels = {
    0: "Sadness",
    1: "Anger",
    2: "Love",
    3: "Surprise",
    4: "Fear",
    5: "Joy"
}


emotion_icons = {
    "Sadness": "😢",
    "Anger": "😠",
    "Love": "❤️",
    "Surprise": "😲",
    "Fear": "😨",
    "Joy": "😄"
}


emotion_descriptions = {
    "Sadness":
        "The text expresses sadness, disappointment, loneliness, or emotional pain.",

    "Anger":
        "The text expresses frustration, irritation, disagreement, or anger.",

    "Love":
        "The text expresses affection, care, attachment, or emotional connection.",

    "Surprise":
        "The text expresses something unexpected, unusual, or shocking.",

    "Fear":
        "The text expresses worry, nervousness, anxiety, or fear.",

    "Joy":
        "The text expresses happiness, excitement, satisfaction, or a positive feeling."
}


# =========================================================
# PREPROCESSING
# =========================================================

def preprocess_text(text):

    text = text.lower()

    text = re.sub(r"\d+", "", text)

    text = text.encode("ascii", "ignore").decode("ascii")

    words = text.split()

    words = [
        word
        for word in words
        if word not in STOP_WORDS
    ]

    return " ".join(words)


# =========================================================
# SESSION STATE
# =========================================================

if "input_text" not in st.session_state:
    st.session_state.input_text = ""


# =========================================================
# CUSTOM UI
# =========================================================

st.html("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap');


/* ========================================================
   GLOBAL
   ======================================================== */

.stApp {

    background:
        radial-gradient(
            circle at 50% -20%,
            rgba(129, 91, 255, 0.10),
            transparent 38%
        ),
        #0b0d12;

    color: #f5f5f7;

    font-family:
        'DM Sans',
        sans-serif;
}


.block-container {

    max-width: 1450px;

    padding-top: 45px;
    padding-bottom: 70px;

}


/* ========================================================
   HERO
   ======================================================== */

.hero {

    text-align: center;

    margin-bottom: 48px;

}


.hero-tag {

    color: #a78bfa;

    font-family:
        'Space Mono',
        monospace;

    font-size: 12px;

    letter-spacing: 0.5px;

    margin-bottom: 18px;

}


.hero-title {

    font-size: 48px;

    line-height: 1;

    font-weight: 700;

    letter-spacing: -2px;

    margin: 0;

    background:
        linear-gradient(
            90deg,
            #ffffff 0%,
            #b99cff 50%,
            #ffffff 100%
        );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;

}


.hero-description {

    margin-top: 22px;

    color: #9ca3af;

    font-size: 15px;

}


.hero-description strong {

    color: #d7d9df;

}


/* ========================================================
   STAT CARDS
   ======================================================== */

.stat-card {

    background: #12151d;

    border: 1px solid #252a35;

    border-radius: 14px;

    padding: 25px 15px;

    text-align: center;

    min-height: 98px;

}


.stat-value {

    color: #f5f5f7;

    font-size: 23px;

    font-weight: 700;

}


.stat-label {

    margin-top: 10px;

    color: #7f8796;

    font-family:
        'Space Mono',
        monospace;

    font-size: 8px;

    letter-spacing: 1px;

    text-transform: uppercase;

}


/* ========================================================
   MAIN CARDS
   ======================================================== */

.main-card {

    background: #12151d;

    border: 1px solid #252a35;

    border-radius: 16px;

    padding: 27px;

}


.card-heading {

    color: #f4f4f5;

    font-size: 17px;

    font-weight: 700;

    margin-bottom: 10px;

}


.card-description {

    color: #858c9b;

    font-size: 12px;

    line-height: 1.6;

    margin-bottom: 18px;

}


/* ========================================================
   TEXT AREA
   ======================================================== */

textarea {

    background: #1b1e27 !important;

    color: #f4f4f5 !important;

    border: 1px solid #292f3b !important;

    border-radius: 10px !important;

    font-size: 14px !important;

}


textarea:focus {

    border-color: #7357e8 !important;

    box-shadow:
        0 0 0 1px #7357e8 !important;

}


/* ========================================================
   ANALYZE BUTTON
   ======================================================== */

.stButton > button {

    background: transparent !important;

    border: 1px solid #3a4050 !important;

    color: #e8e9ed !important;

    border-radius: 8px !important;

    height: 42px !important;

    font-size: 13px !important;

    font-weight: 500 !important;

    transition: all 0.2s ease;

}


.stButton > button:hover {

    border-color: #8b72f2 !important;

    background: rgba(
        139,
        114,
        242,
        0.08
    ) !important;

}


/* ========================================================
   EXAMPLE BUTTONS
   ======================================================== */

.example-button {

    background: #10131a;

    border: 1px solid #292f3b;

    border-radius: 9px;

    padding: 12px 13px;

    margin-bottom: 9px;

    color: #cfd2da;

    font-size: 11px;

}


/* ========================================================
   PREDICTION
   ======================================================== */

.result-card {

    margin-top: 25px;

    background:
        linear-gradient(
            135deg,
            rgba(116, 87, 235, 0.14),
            rgba(22, 25, 34, 0.95)
        );

    border: 1px solid rgba(
        115,
        87,
        235,
        0.35
    );

    border-radius: 16px;

    padding: 34px;

    text-align: center;

}


.result-icon {

    font-size: 50px;

}


.result-label {

    color: #898fa0;

    font-family:
        'Space Mono',
        monospace;

    font-size: 9px;

    letter-spacing: 1.5px;

    text-transform: uppercase;

    margin-top: 12px;

}


.result-emotion {

    color: #ffffff;

    font-size: 32px;

    font-weight: 700;

    margin-top: 8px;

}


/* ========================================================
   PROBABILITY
   ======================================================== */

.probability-card {

    margin-top: 25px;

    background: #12151d;

    border: 1px solid #252a35;

    border-radius: 16px;

    padding: 27px;

}


.probability-title {

    color: #f3f4f6;

    font-size: 17px;

    font-weight: 700;

}


.probability-subtitle {

    color: #858c9b;

    font-size: 12px;

    margin-top: 7px;

}


/* ========================================================
   FOOTER
   ======================================================== */

.footer {

    text-align: center;

    color: #5f6674;

    font-family:
        'Space Mono',
        monospace;

    font-size: 9px;

    letter-spacing: 0.5px;

    margin-top: 55px;

}


</style>
""")


# =========================================================
# HERO
# =========================================================

st.html("""
<div class="hero">

    <div class="hero-tag">
        ✦ NLP · MACHINE LEARNING · TEXT ANALYSIS
    </div>

    <div class="hero-title">
        EmotiSense
    </div>

    <div class="hero-description">
        Understand the emotion behind your words using
        <strong>Natural Language Processing</strong>
        and <strong>Machine Learning</strong>.
    </div>

</div>
""")


# =========================================================
# STATISTICS
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.html("""
    <div class="stat-card">

        <div class="stat-value">
            86%
        </div>

        <div class="stat-label">
            Model Accuracy
        </div>

    </div>
    """)


with col2:

    st.html("""
    <div class="stat-card">

        <div class="stat-value">
            6
        </div>

        <div class="stat-label">
            Emotion Classes
        </div>

    </div>
    """)


with col3:

    st.html("""
    <div class="stat-card">

        <div class="stat-value">
            16K
        </div>

        <div class="stat-label">
            Training Samples
        </div>

    </div>
    """)


with col4:

    st.html("""
    <div class="stat-card">

        <div class="stat-value">
            TF-IDF
        </div>

        <div class="stat-label">
            Text Representation
        </div>

    </div>
    """)


# =========================================================
# SPACE
# =========================================================

st.write("")


# =========================================================
# MAIN INPUT + EXAMPLES
# =========================================================

left, right = st.columns(
    [1.65, 1],
    gap="large"
)


# =========================================================
# LEFT
# =========================================================

with left:

    st.html("""
    <div class="main-card">

        <div class="card-heading">
            Analyze a sentence
        </div>

        <div class="card-description">
            Enter any sentence and the NLP model will identify
            the most likely emotion.
        </div>

    </div>
    """)


    user_text = st.text_area(
        "Input",
        value=st.session_state.input_text,
        height=155,
        placeholder="Type something like: I feel really happy today!",
        label_visibility="collapsed"
    )


    analyze_button = st.button(
        "✦ Analyze Emotion",
        use_container_width=True
    )


# =========================================================
# RIGHT
# =========================================================

with right:

    st.html("""
    <div class="main-card">

        <div class="card-heading">
            Try an example
        </div>

        <div class="card-description">
            Test the model with different emotions.
        </div>

    </div>
    """)


    examples = [
        ("😄", "I'm so happy that everything worked out."),
        ("😠", "This situation is really frustrating me."),
        ("😢", "I feel lonely and disappointed today."),
        ("😨", "I'm worried about what might happen.")
    ]


    for icon, example in examples:

        if st.button(
            f"{icon}  {example}",
            key=f"example_{example}",
            use_container_width=True
        ):

            st.session_state.input_text = example

            st.rerun()


# =========================================================
# ANALYZE
# =========================================================

if analyze_button:

    if not user_text.strip():

        st.warning(
            "Please enter a sentence first."
        )

    else:

        # -------------------------------------------------
        # PREPROCESS
        # -------------------------------------------------

        cleaned_text = preprocess_text(
            user_text
        )


        # -------------------------------------------------
        # TF-IDF
        # -------------------------------------------------

        text_vector = tfidf_vectorizer.transform(
            [cleaned_text]
        )


        # -------------------------------------------------
        # MODEL PREDICTION
        # -------------------------------------------------

        prediction = model.predict(
            text_vector
        )[0]

        prediction = int(prediction)


        predicted_emotion = emotion_labels.get(
            prediction,
            str(prediction)
        )


        # -------------------------------------------------
        # PROBABILITIES
        # -------------------------------------------------

        probabilities = model.predict_proba(
            text_vector
        )[0]


        # =================================================
        # RESULT
        # =================================================

        icon = emotion_icons.get(
            predicted_emotion,
            "🧠"
        )


        st.html(f"""
        <div class="result-card">

            <div class="result-icon">
                {icon}
            </div>

            <div class="result-label">
                Detected Emotion
            </div>

            <div class="result-emotion">
                {predicted_emotion}
            </div>

        </div>
        """)


        # =================================================
        # DESCRIPTION
        # =================================================

        st.html(f"""
        <div class="main-card" style="margin-top:18px;">

            <div class="card-heading">
                What this means
            </div>

            <div class="card-description">
                {emotion_descriptions.get(
                    predicted_emotion,
                    "Emotion predicted by the model."
                )}
            </div>

        </div>
        """)


        # =================================================
        # PROBABILITY DATA
        # =================================================

        probability_data = []


        for class_index, probability in zip(
            model.classes_,
            probabilities
        ):

            class_index = int(
                class_index
            )


            probability_data.append({

                "Emotion":
                    emotion_labels.get(
                        class_index,
                        str(class_index)
                    ),

                "Probability":
                    float(probability)

            })


        probability_df = pd.DataFrame(
            probability_data
        )


        # =================================================
        # PROBABILITY HEADER
        # =================================================

        st.html("""
        <div class="probability-card">

            <div class="probability-title">
                Emotion Probability
            </div>

            <div class="probability-subtitle">
                Probability distribution generated by
                the Logistic Regression classifier.
            </div>

        </div>
        """)


        # =================================================
        # SAFE CHART
        # =================================================

        chart_df = probability_df.set_index(
            "Emotion"
        )


        st.bar_chart(
            chart_df[
                "Probability"
            ],
            height=300
        )


        # =================================================
        # PROBABILITY TABLE
        # =================================================

        display_df = probability_df.copy()


        display_df["Probability"] = (
            display_df["Probability"] * 100
        ).round(2)


        display_df = display_df.sort_values(
            "Probability",
            ascending=False
        )


        display_df["Probability"] = (
            display_df["Probability"]
            .astype(str)
            + "%"
        )


        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True
        )


        # =================================================
        # MODEL EXPLANATION
        # =================================================

        with st.expander(
            "How the model works"
        ):

            st.markdown("""
### Prediction Pipeline

**1. Text Preprocessing**

The input text is converted to lowercase,
numbers and non-ASCII characters are removed,
and English stopwords are removed.

**2. TF-IDF**

The processed text is converted into numerical
features using the fitted TF-IDF vectorizer.

**3. Logistic Regression**

The TF-IDF features are passed to the trained
Logistic Regression classifier.

**4. Emotion Prediction**

The model predicts one of six emotions:

- Sadness
- Anger
- Love
- Surprise
- Fear
- Joy

**5. Probability Distribution**

The classifier's probability distribution is
shown below the prediction.
""")


# =========================================================
# FOOTER
# =========================================================

st.html("""
<div class="footer">
    EMOTISENSE · NLP EMOTION CLASSIFICATION
</div>
""")
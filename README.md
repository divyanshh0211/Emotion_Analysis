## 🚀 Live Demo

Try the deployed application:

**[Open Emotion Analysis App](https://divyanshh0211-emotion-analysis-app-ipjneo.streamlit.app/)**

The application allows users to enter text and receive an emotion prediction along with the prediction probability distribution.

# Emotion Analysis using NLP

> An end-to-end Natural Language Processing project that identifies emotions expressed in text using TF-IDF and Logistic Regression, with an interactive Streamlit interface for real-time predictions.

---

## 📌 About the Project

People express emotions differently through text, and identifying those emotions automatically is a useful NLP problem with applications in customer feedback, social media analysis, conversational systems, and user behavior analysis.

This project explores how traditional machine learning techniques can be used to build an emotion classification system without relying on large language models or pre-trained transformer architectures.

The system takes a text sentence as input, processes it using a consistent NLP preprocessing pipeline, converts it into numerical features using TF-IDF, and then uses a trained Logistic Regression classifier to predict the emotion.

The final model is integrated into a Streamlit application so that the trained model can be used interactively.

---

## 🎯 Objective

The main objective of this project was to build a complete NLP classification pipeline and understand each stage involved in converting raw text into a machine learning prediction.

The project focuses on:

- Cleaning and preprocessing raw text
- Converting text into meaningful numerical features
- Training and comparing machine learning models
- Selecting a suitable classification algorithm
- Saving the trained model for reuse
- Building an interactive interface for predictions

---

## 💡 What Can It Detect?

The model predicts one of six emotions:

| Emotion | Example |
|---|---|
| 😢 Sadness | "I feel completely alone today." |
| 😠 Anger | "This situation is really frustrating." |
| ❤️ Love | "I really care about you." |
| 😲 Surprise | "I didn't expect that to happen!" |
| 😨 Fear | "I am worried about what might happen." |
| 😄 Joy | "Today has been an amazing day!" |

---

## 🔄 End-to-End Workflow

The complete project follows this pipeline:

```text
                    ┌─────────────────┐
                    │    Text Input   │
                    └────────┬────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Text Preprocessing  │
                  │                     │
                  │ • Lowercase         │
                  │ • Remove Numbers    │
                  │ • Remove Non-ASCII  │
                  │ • Stopword Removal  │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │   TF-IDF Vectorizer │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Logistic Regression │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Emotion Prediction  │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Streamlit Interface │
                  └─────────────────────┘

```

---

## 📊 Dataset

The project uses a dataset of **16,000 text samples** covering **6 different emotion classes**. Each sample contains a piece of text along with its corresponding emotion label.

### Dataset Overview

| Property | Details |
|---|---|
| Total Samples | 16,000 |
| Number of Classes | 6 |
| Input | Text |
| Output | Emotion Label |
| Problem Type | Multi-Class Classification |
| Feature Extraction | TF-IDF |
| Final Classifier | Logistic Regression |

### Emotion Classes

| Label | Emotion |
|:---:|---|
| 0 | Sadness |
| 1 | Anger |
| 2 | Love |
| 3 | Surprise |
| 4 | Fear |
| 5 | Joy |

The dataset provides the text that the model learns from, with each sentence associated with one of the six emotion categories.

---

## 🔬 What I Did

I built the project step by step, starting with the raw text data and ending with a working application.

### 1. Data Preparation

The dataset was first loaded and explored to understand the text samples and their corresponding emotion labels.

The main objective at this stage was to prepare the text in a form that could be used effectively for machine learning.

### 2. Text Preprocessing

Raw text contains variations that are not always useful for classification. I applied a simple preprocessing pipeline to make the text more consistent.

The preprocessing steps include:

- Converting text to lowercase
- Removing numbers
- Removing non-ASCII characters
- Removing English stopwords using NLTK

The same preprocessing approach is used when a user enters new text in the application.

### 3. Converting Text into Numbers

Machine learning models work with numerical data, so the cleaned text was converted into numerical feature vectors using **TF-IDF**.

TF-IDF was used to represent the importance of words within the text while reducing the influence of very common words.

The fitted vectorizer was saved for use during prediction:

```text
tfidf_vectorizer.pkl

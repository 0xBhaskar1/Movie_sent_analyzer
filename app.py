
import streamlit as st
import joblib
import json
import os

import pandas as pd

from text_preprocessing import clean_text



# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="wide"
)


# -----------------------------------
# CUSTOM CSS
# -----------------------------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #101827, #172033);
    color: white;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    color: #ffffff;
}

.subtitle {
    text-align: center;
    color: #aab5c5;
    font-size: 17px;
    margin-bottom: 30px;
}

.result-positive {
    background-color: #123c2b;
    border: 1px solid #22c55e;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    color: #4ade80;
    font-size: 26px;
    font-weight: bold;
}

.result-negative {
    background-color: #451c25;
    border: 1px solid #ef4444;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    color: #f87171;
    font-size: 26px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------------
# LOAD TRAINED MODELS
# -----------------------------------

@st.cache_resource
def load_artifacts():

    vectorizer = joblib.load(
        "model_artifacts/vectorizer.pkl"
    )

    models = joblib.load(
        "model_artifacts/models.pkl"
    )

    with open("model_artifacts/metrics.json", "r") as file:
        metrics = json.load(file)

    return vectorizer, models, metrics


# -----------------------------------
# APPLICATION HEADER
# -----------------------------------

st.markdown(
    '<div class="main-title">🎬 Movie Sentiment Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Discover whether a movie review is Positive or Negative using Machine Learning.</div>',
    unsafe_allow_html=True
)


# -----------------------------------
# LOAD ARTIFACTS
# -----------------------------------

try:

    vectorizer, models, metrics = load_artifacts()

except FileNotFoundError:

    st.error(
        "Model files not found. Please execute python train.py first."
    )

    st.stop()


# -----------------------------------
# SIDEBAR
# -----------------------------------

st.sidebar.title("⚙️ Settings")

selected_model = st.sidebar.selectbox(
    "Choose Machine Learning Algorithm",
    list(models.keys())
)

st.sidebar.markdown("---")

st.sidebar.info(
    "This application uses the IMDb movie review dataset "
    "to classify reviews into positive and negative sentiments."
)


# -----------------------------------
# USER INPUT
# -----------------------------------

st.subheader("Enter Your Movie Review")

review = st.text_area(
    "Write your review below:",
    placeholder="Example: The movie was absolutely fantastic. "
                "The acting was brilliant and the story was amazing...",
    height=180
)

predict_button = st.button(
    "🔍 Analyze Sentiment",
    use_container_width=True
)


# -----------------------------------
# PREDICTION
# -----------------------------------

if predict_button:

    if not review.strip():

        st.warning("Please enter a movie review first.")

    else:

        cleaned_review = clean_text(review)

        review_vector = vectorizer.transform(
            [cleaned_review]
        )

        model = models[selected_model]

        prediction = model.predict(review_vector)[0]

        st.markdown("---")

        st.subheader("Prediction Result")

        if prediction == 1:

            st.markdown(
                '<div class="result-positive">'
                '😊 Positive Sentiment'
                '</div>',
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                '<div class="result-negative">'
                '😞 Negative Sentiment'
                '</div>',
                unsafe_allow_html=True
            )

        st.write("")

        st.write(f"**Algorithm Used:** {selected_model}")

        # Probability is available for NB and Logistic Regression.
        # LinearSVC does not provide predict_proba by default.

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(review_vector)[0]

            negative_probability = probabilities[0] * 100
            positive_probability = probabilities[1] * 100

            st.write("### Prediction Scores")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Negative score",
                    f"{negative_probability:.2f}%"
                )

            with col2:
                st.metric(
                    "Positive score",
                    f"{positive_probability:.2f}%"
                )

            st.caption(
                "These are model-estimated class probabilities, "
                "not guaranteed measures of prediction correctness."
            )

        else:

            margin = model.decision_function(review_vector)[0]

            st.metric(
                "SVM Decision Score",
                f"{margin:.4f}"
            )

            st.caption(
                "A positive score indicates positive sentiment; "
                "a negative score indicates negative sentiment. "
                "The score is not a probability."
            )


# -----------------------------------
# MODEL PERFORMANCE
# -----------------------------------

st.markdown("---")

st.header("📊 Model Performance Comparison")

st.write(
    "The following metrics were calculated using the "
    "separate IMDb testing dataset."
)

metrics_df = pd.DataFrame(metrics).T

st.dataframe(
    metrics_df.style.format("{:.2%}"),
    use_container_width=True
)


# -----------------------------------
# ACCURACY COMPARISON
# -----------------------------------

st.subheader("Accuracy Comparison")

accuracy_data = {
    model_name: values["accuracy"] * 100
    for model_name, values in metrics.items()
}

accuracy_df = pd.DataFrame(
    list(accuracy_data.items()),
    columns=["Algorithm", "Accuracy"]
)

st.bar_chart(
    accuracy_df,
    x="Algorithm",
    y="Accuracy"
)


# -----------------------------------
# FOOTER
# -----------------------------------

st.markdown("---")

st.markdown(
    "<center>Built using Python | Scikit-learn | NLP | Streamlit"
    "</center>",
    unsafe_allow_html=True
)

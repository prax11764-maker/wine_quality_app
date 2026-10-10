"""
app.py

A small web app for the Wine Quality Prediction project.
Lets anyone enter a wine's 11 physicochemical properties and get a live
predicted quality score (3-8), using the model trained in train_and_save.py.

Run with:  streamlit run app.py
(Run train_and_save.py first so the .pkl files exist.)
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="centered")

# ---------- Load the trained model + preprocessing objects ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load("wine_model.pkl")
    scaler = joblib.load("wine_scaler.pkl")
    le = joblib.load("wine_label_encoder.pkl")
    feature_names = joblib.load("wine_feature_names.pkl")
    return model, scaler, le, feature_names

model, scaler, le, feature_names = load_artifacts()

# ---------- Header ----------
st.title("🍷 Wine Quality Predictor")
st.write(
    "Enter a wine's physicochemical properties below and get a predicted "
    "quality score (3 = poor, 8 = excellent), based on a Random Forest model "
    "trained on 1,599 red wine samples."
)

st.divider()

# ---------- Input form ----------
# (min, max, default) chosen from the real dataset's range for each feature,
# so sliders stay realistic instead of allowing nonsense values.
feature_ranges = {
    "fixed acidity":        (4.0, 16.0, 8.3, 0.1),
    "volatile acidity":     (0.1, 1.6, 0.53, 0.01),
    "citric acid":          (0.0, 1.0, 0.27, 0.01),
    "residual sugar":       (0.5, 16.0, 2.5, 0.1),
    "chlorides":            (0.01, 0.62, 0.087, 0.001),
    "free sulfur dioxide":  (1.0, 72.0, 15.9, 1.0),
    "total sulfur dioxide": (6.0, 290.0, 46.5, 1.0),
    "density":              (0.990, 1.004, 0.9967, 0.0001),
    "pH":                   (2.7, 4.1, 3.31, 0.01),
    "sulphates":            (0.3, 2.0, 0.66, 0.01),
    "alcohol":              (8.0, 15.0, 10.4, 0.1),
}

st.subheader("Wine Properties")
col1, col2 = st.columns(2)
user_input = {}

for i, feature in enumerate(feature_names):
    lo, hi, default, step = feature_ranges[feature]
    target_col = col1 if i % 2 == 0 else col2
    user_input[feature] = target_col.slider(
        feature.title(), min_value=float(lo), max_value=float(hi),
        value=float(default), step=float(step)
    )

st.divider()

# ---------- Predict ----------
if st.button("Predict Quality", type="primary"):
    sample = pd.DataFrame([user_input], columns=feature_names)
    sample_scaled = scaler.transform(sample)

    pred_encoded = model.predict(sample_scaled)[0]
    pred_score = le.inverse_transform([pred_encoded])[0]

    # Get probability for every possible quality score, for a fuller picture
    proba = model.predict_proba(sample_scaled)[0]
    proba_df = pd.DataFrame({
        "Quality Score": le.classes_,
        "Confidence": proba
    }).sort_values("Quality Score")

    st.subheader("Result")
    st.metric("Predicted Quality Score", f"{pred_score} / 8")

    if pred_score >= 7:
        st.success("This wine is predicted to be excellent quality! 🌟")
    elif pred_score >= 6:
        st.info("This wine is predicted to be good/above-average quality.")
    else:
        st.warning("This wine is predicted to be below-average quality.")

    st.write("Model confidence across all possible scores:")
    st.bar_chart(proba_df.set_index("Quality Score"))

    with st.expander("See the exact values used for this prediction"):
        st.dataframe(sample.T.rename(columns={0: "Value"}))

st.divider()
st.caption(
    "Model: Random Forest (class-balanced) | Dataset: UCI Wine Quality (red wine) | "
    "Built by Prajna Paramita Jena"
)

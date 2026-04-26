"""Streamlit app: Airline Passenger Satisfaction predictor.

User enters a passenger profile, the model returns a satisfaction
probability and a SHAP waterfall plot explaining the prediction.

Run locally:
    streamlit run app.py

Deploy to Streamlit Community Cloud:
    1. Push this folder to a public GitHub repo
    2. Go to share.streamlit.io and connect the repo
    3. Set main file path to app.py
"""

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap

st.set_page_config(
    page_title="Airline Passenger Satisfaction",
    page_icon="✈",
    layout="wide",
)

# ─── Load saved model and metadata ────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    pipe = joblib.load('pipeline.joblib')
    with open('feature_meta.json') as f:
        meta = json.load(f)
    return pipe, meta

pipeline, meta = load_artifacts()

# ─── Header ───────────────────────────────────────────────────────────────
st.title("Airline Passenger Satisfaction Predictor")
st.markdown(
    "Predicts whether a passenger will be **satisfied** or "
    "**neutral / dissatisfied** based on their travel profile and "
    "in-flight service ratings. Built on the Kaggle Airline Passenger "
    "Satisfaction dataset (~130k passengers). Model: XGBoost inside an sklearn Pipeline."
)

# ─── Sidebar input form ───────────────────────────────────────────────────
st.sidebar.header("Passenger profile")

# Service ratings (Likert 0-5)
SERVICE_RATINGS = [
    'Inflight wifi service', 'Departure/Arrival time convenient',
    'Ease of Online booking', 'Gate location', 'Food and drink',
    'Online boarding', 'Seat comfort', 'Inflight entertainment',
    'On-board service', 'Leg room service', 'Baggage handling',
    'Checkin service', 'Inflight service', 'Cleanliness',
]

# Categorical inputs
gender        = st.sidebar.selectbox("Gender", meta['cat_levels'].get('Gender', ['Male', 'Female']))
customer_type = st.sidebar.selectbox("Customer Type", meta['cat_levels'].get('Customer Type', ['Loyal Customer', 'disloyal Customer']))
travel_type   = st.sidebar.selectbox("Type of Travel", meta['cat_levels'].get('Type of Travel', ['Business travel', 'Personal Travel']))
travel_class  = st.sidebar.selectbox("Class", meta['cat_levels'].get('Class', ['Eco', 'Eco Plus', 'Business']))

# Numeric inputs
age = st.sidebar.slider("Age", 7, 85, 35)
flight_distance = st.sidebar.slider(
    "Flight Distance (miles)",
    int(meta['num_ranges']['Flight Distance']['min']),
    int(meta['num_ranges']['Flight Distance']['max']),
    int(meta['num_ranges']['Flight Distance']['median']),
)

st.sidebar.subheader("Service ratings (0 = N/A, 1 = poor, 5 = excellent)")
ratings = {}
for feat in SERVICE_RATINGS:
    if feat in meta['num_cols']:
        ratings[feat] = st.sidebar.slider(feat, 0, 5, 3)

# ─── Build a single-row DataFrame matching the training schema ─────────────
def build_input_row():
    row = {
        'Gender':        gender,
        'Customer Type': customer_type,
        'Type of Travel': travel_type,
        'Class':         travel_class,
        'Age':           age,
        'Flight Distance': flight_distance,
        # Engineered Week 2 features
        'Flight Distance Log': float(np.log1p(flight_distance)),
        'Class Ordinal':       {'Eco': 0, 'Eco Plus': 1, 'Business': 2}[travel_class],
        'Distance_Class':      flight_distance * {'Eco': 0, 'Eco Plus': 1, 'Business': 2}[travel_class],
    }
    row.update(ratings)
    # Fill any other expected columns with median / mode
    for c in meta['num_cols']:
        row.setdefault(c, meta['num_ranges'][c]['median'])
    for c in meta['cat_cols']:
        row.setdefault(c, meta['cat_levels'][c][0])
    # Re-order columns to match the pipeline expectation
    cols = meta['num_cols'] + meta['cat_cols']
    return pd.DataFrame([row])[cols]

X_one = build_input_row()

# ─── Predict ──────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Prediction")
    proba = pipeline.predict_proba(X_one)[0, 1]
    label = "Satisfied" if proba >= 0.5 else "Neutral / Dissatisfied"
    color = "#1F8A4C" if proba >= 0.5 else "#C03A2B"
    st.markdown(
        f"<div style='padding:1rem;border-radius:8px;background:{color};color:white;'>"
        f"<h2 style='margin:0;color:white;'>{label}</h2>"
        f"<p style='margin:0;font-size:1.2rem;'>Confidence: {proba*100:.1f}% satisfied</p>"
        f"</div>",
        unsafe_allow_html=True
    )
    st.progress(float(proba))

# ─── SHAP waterfall explanation ───────────────────────────────────────────
with col2:
    st.subheader("Why this prediction? (SHAP waterfall)")

    prep = pipeline.named_steps['pre']
    clf  = pipeline.named_steps['clf']
    num_names = meta['num_cols']
    cat_names = prep.named_transformers_['cat']['enc'].get_feature_names_out(meta['cat_cols']).tolist()
    feat_names = num_names + cat_names

    X_one_prep = pd.DataFrame(prep.transform(X_one), columns=feat_names)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer(X_one_prep)

    fig = plt.figure(figsize=(8, 6))
    shap.plots.waterfall(shap_values[0], max_display=12, show=False)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    st.markdown(
        "**How to read this:** Red bars push the prediction toward "
        "*satisfied*; blue bars push toward *dissatisfied*. The sum of "
        "all bar lengths equals the model's final score above."
    )

# ─── Footer ───────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Group 5 — MSc Machine Learning | Aidar Batyrbekov, Zhanerke Ismagulova, Timur Baitukenov | "
    "Dataset: Kaggle teejmahal20/airline-passenger-satisfaction"
)

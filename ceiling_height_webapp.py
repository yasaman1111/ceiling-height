
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

st.set_page_config(page_title="Children's Ceiling Perception", layout="centered")

st.title("🏠 Predicting Children's Perception of Ceiling Height")
st.markdown(
    "This web app uses VR experiment data to predict how children (ages 6–8) perceive ceiling heights "
    "in different architectural spaces. Based on machine learning (ANN) trained on your dataset."
)

# ---------- Load the dataset ----------
df = pd.read_csv("ceiling_height_children.csv")

# ---------- Train/Test Split ----------
X = df[["Space", "Ceiling_Height"]]
y = df["Label"]
classes = ["Very Short", "Short", "Tall", "Very Tall"]

# ---------- Preprocessing pipeline ----------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["Ceiling_Height"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["Space"]),
    ]
)

model = Pipeline(
    steps=[
        ("prep", preprocessor),
        ("clf", MLPClassifier(hidden_layer_sizes=(10, 8), max_iter=1000, random_state=42)),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
model.fit(X_train, y_train)

# ---------- Sidebar Input ----------
st.sidebar.header("Input Parameters")
space_input = st.sidebar.selectbox("Select Space Type", sorted(df["Space"].unique()))
min_h = df[df["Space"] == space_input]["Ceiling_Height"].min()
max_h = df[df["Space"] == space_input]["Ceiling_Height"].max()
height_input = st.sidebar.slider("Ceiling Height (m)", float(min_h), float(max_h), float(round((min_h + max_h) / 2, 2)), 0.01)

# ---------- Prediction ----------
input_df = pd.DataFrame([[space_input, height_input]], columns=["Space", "Ceiling_Height"])
pred_probs = model.predict_proba(input_df)[0]
pred_df = pd.DataFrame({"Perception": classes, "Probability (%)": (pred_probs * 100).round(2)}).set_index("Perception")

st.subheader("Prediction Probabilities")
st.bar_chart(pred_df)

# ---------- Model performance (optional) ----------
with st.expander("Model Performance (on test set)"):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=classes)
    st.text(report)

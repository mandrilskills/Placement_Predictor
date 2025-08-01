import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
st.markdown("""
<style>
/* Light soothing background */
.stApp {
    background-color: #f5f3ff;  /* light lavender */
    color: #333333;
    font-family: 'Segoe UI', sans-serif;
}

/* Center container with soft card styling */
div.block-container {
    padding: 2rem;
    max-width: 850px;
    margin: auto;
    background-color: #ffffff;
    border-radius: 15px;
    box-shadow: 0 0 20px rgba(0,0,0,0.05);
}

/* Input boxes */
input, .stNumberInput input {
    background-color: #f0f0ff !important;
    color: #333333 !important;
    border-radius: 10px !important;
}

/* Sliders */
div[data-baseweb="slider"] > div {
    background: #cbd5f5 !important;
}
span[data-testid="stSliderValue"] {
    color: #4b0082 !important;
}

/* Buttons */
.stButton > button {
    background-color: #c3b5f5;
    color: black;
    font-weight: 600;
    border-radius: 8px;
    padding: 0.4rem 1rem;
}

/* Headings and text */
h1, h2, h3, .stMarkdown {
    color: #4b0082;
}

.css-1emrehy.edgvbvh3 { 
    color: #4b0082;
}
</style>
""", unsafe_allow_html=True)


df = pd.read_csv('Placement.csv')
X = df.drop('placement', axis=1)
y = df['placement']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

st.title("🎯 Your Placement Predictor")
st.caption("Developed by Mandril Sircar")
st.markdown("Fill in the student details below to predict the likelihood of placement.")
sgpa = st.slider("Average SGPA (1.0 to 10.0)", 1.0, 10.0, step=0.1)
intern = st.number_input("Number of Internships", min_value=0, max_value=10, value=0)
backlogs = st.number_input("Number of Backlogs", min_value=0, max_value=10, value=0)
communication_skill = st.slider("Communication Skill (1 to 10)", 1, 10, value=7)
aptitude_score = st.slider("Aptitude Score (0 to 100)", 0, 100, value=70)
projects = st.number_input("Number of Major Projects", min_value=0, max_value=10, value=1)
certifications = st.number_input("Number of Certifications", min_value=0, max_value=10, value=1)
extra_curricular = st.selectbox("Active in Extra Curricular Activities?", options=[0, 1])
if st.button("Predict Placement"):
    student_input = np.array([[sgpa, intern, backlogs, communication_skill,
                               aptitude_score, projects, certifications, extra_curricular]])
    prediction = model.predict(student_input)[0]
    prob = model.predict_proba(student_input)[0][1]

    st.markdown("---")
    if prediction == 1:
        st.success("The student is **likely to be placed.**")
    else:
        st.error("The student is **unlikely to be placed.**")

    st.write(f"**Probability of Placement:** `{prob * 100:.2f}%`")

if st.checkbox("Show Model Accuracy on Test Data"):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write("Model Accuracy:", f"{acc * 100:.2f}%")

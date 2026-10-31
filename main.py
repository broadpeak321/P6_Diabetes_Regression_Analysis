
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Diabetes Prediction AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #eef2ff, #f0fdfa);
}

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #312e81;
    text-align: center;
}

.subtitle {
    text-align: center;
    color: #64748b;
    font-size: 17px;
    margin-bottom: 25px;
}

.metric-card {
    background: white;
    padding: 22px;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.07);
    text-align: center;
    border-top: 5px solid #6366f1;
}

.metric-value {
    font-size: 30px;
    font-weight: bold;
    color: #312e81;
}

.metric-label {
    font-size: 14px;
    color: #64748b;
}

.section-title {
    color: #3730a3;
    font-size: 23px;
    font-weight: 700;
}

div.stButton > button {
    background: linear-gradient(90deg, #4f46e5, #06b6d4);
    color: white;
    border-radius: 10px;
    height: 48px;
    font-size: 17px;
    font-weight: bold;
    border: none;
    width: 100%;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #4338ca, #0891b2);
    color: white;
}

[data-testid="stSidebar"] {
    background-color: #e0e7ff;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DATASET AND MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    model = joblib.load("best_diabetes_model.joblib")

    return model

@st.cache_data
def load_data():

    diabetes = load_diabetes(as_frame=True)

    return diabetes.frame.drop(columns="target")

try:
    model = load_model()
    X = load_data()

except Exception as e:
    st.error(f"Error loading model or dataset: {e}")
    st.stop()

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🩺 Diabetes Progression Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Based Diabetes Progression Estimation'
    '</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">10</div>
        <div class="metric-label">Input Features</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">14</div>
        <div class="metric-label">Regressors Compared</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">5-Fold</div>
        <div class="metric-label">Cross Validation</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# --------------------------------------------------
# FEATURE INFORMATION
# --------------------------------------------------

feature_info = {
    "age": "Age of the patient",
    "sex": "Sex indicator",
    "bmi": "Body Mass Index",
    "bp": "Average blood pressure",
    "s1": "Total serum cholesterol",
    "s2": "Low-density lipoproteins",
    "s3": "High-density lipoproteins",
    "s4": "Total cholesterol / HDL ratio",
    "s5": "Log of serum triglycerides",
    "s6": "Blood sugar level"
}

# --------------------------------------------------
# SIDEBAR - USER INPUTS
# --------------------------------------------------

st.sidebar.markdown("## 🎛️ Patient Parameters")

st.sidebar.info(
    "Adjust the sliders to enter patient parameters "
    "and predict the diabetes progression score."
)

user_values = {}

for feature in X.columns:

    minimum = float(X[feature].min())
    maximum = float(X[feature].max())
    default = float(X[feature].median())

    step = (maximum - minimum) / 200

    user_values[feature] = st.sidebar.slider(
        label=f"{feature.upper()}",
        min_value=minimum,
        max_value=maximum,
        value=default,
        step=step,
        help=feature_info[feature]
    )

if st.sidebar.button("🔄 Reset Parameters"):

    for feature in X.columns:
        st.session_state.pop(feature, None)

    st.rerun()

# --------------------------------------------------
# MAIN DASHBOARD
# --------------------------------------------------

left, right = st.columns([1.2, 1])

with left:

    st.markdown(
        '<div class="section-title">👤 Patient Input Summary</div>',
        unsafe_allow_html=True
    )

    input_df = pd.DataFrame(
        [user_values],
        columns=X.columns
    )

    st.dataframe(
        input_df.T.rename(columns={0: "Input Value"}),
        use_container_width=True
    )

with right:

    st.markdown(
        '<div class="section-title">📊 Input Visualization</div>',
        unsafe_allow_html=True
    )

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.barh(
        list(user_values.keys()),
        list(user_values.values())
    )

    ax.set_xlabel("Feature Value")
    ax.set_title("Patient Feature Distribution")

    ax.grid(axis="x", alpha=0.2)

    fig.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

st.write("---")

st.markdown(
    '<div class="section-title">🤖 AI Prediction</div>',
    unsafe_allow_html=True
)

st.write(
    "Click the button below to estimate the diabetes "
    "progression score using the trained machine learning model."
)

if st.button("🚀 Predict Diabetes Progression"):

    prediction = model.predict(input_df)[0]

    st.markdown("### Prediction Result")

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(90deg,#4f46e5,#06b6d4);
            padding: 30px;
            border-radius: 18px;
            text-align: center;
            color: white;
        ">
            <div style="font-size:18px;">
                Predicted Progression Score
            </div>
            <div style="font-size:45px;font-weight:800;">
                {prediction:.2f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.success("Prediction generated successfully!")

    st.caption(
        "This is an educational machine learning estimate, "
        "not a clinical diagnosis or an individual medical risk score."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.write("---")

st.markdown(
    """
    <div style="text-align:center;color:#64748b;">
        Diabetes Progression Prediction | 
        Scikit-learn | Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True
)

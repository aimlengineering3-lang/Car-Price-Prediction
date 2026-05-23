import streamlit as st
import pandas as pd
import numpy as np
import pickle
from lightgbm import LGBMRegressor

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="OLX Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 OLX Car Price Predictor")

# ---------------- LOAD MODEL ---------------- #

model = pickle.load(open("models/lightgbm_model.pkl", "rb"))

le_make = pickle.load(open("models/le_make.pkl", "rb"))
le_model = pickle.load(open("models/le_model.pkl", "rb"))
le_fuel = pickle.load(open("models/le_fuel.pkl", "rb"))
le_city = pickle.load(open("models/le_city.pkl", "rb"))

# ---------------- CUSTOM CSS ---------------- #

st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

.title {
    font-size: 50px;
    font-weight: bold;
    color: #00FFAA;
    text-align: center;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    color: #AAAAAA;
}

.card {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 0px 15px rgba(0,255,170,0.2);
}

.prediction {
    font-size: 40px;
    color: #00FFAA;
    font-weight: bold;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ---------------- #

st.markdown('<p class="title">🚗 OLX Car Price Predictor</p>', unsafe_allow_html=True)

st.markdown(
    '<p class="subtitle">Predict Used Car Prices Using LightGBM Machine Learning</p>',
    unsafe_allow_html=True
)

st.write("")

# ---------------- LAYOUT ---------------- #

col1, col2 = st.columns([1,1])

with col1:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    make = st.selectbox(
        "Select Make",
        le_make.classes_
    )

    model_name = st.selectbox(
        "Select Model",
        le_model.classes_
    )

    year = st.slider(
        "Select Year",
        1990,
        2026,
        2018
    )

    kms = st.number_input(
        "KM Driven",
        min_value=0,
        value=50000
    )

    fuel = st.selectbox(
        "Fuel Type",
        le_fuel.classes_
    )

    city = st.selectbox(
        "Registration City",
        le_city.classes_
    )

    st.markdown('</div>', unsafe_allow_html=True)

with col2:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.image(
        "https://images.unsplash.com/photo-1503376780353-7e6692767b70",
        use_container_width=True
    )

    st.markdown("""
    ### Why This App?
    
    - LightGBM Machine Learning
    - Real-Time Prediction
    - Advanced Feature Engineering
    - Modern Streamlit UI
    - Fast & Accurate
    
    """)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PREDICTION ---------------- #

if st.button("Predict Price"):

    make_encoded = le_make.transform([make])[0]
    model_encoded = le_model.transform([model_name])[0]
    fuel_encoded = le_fuel.transform([fuel])[0]
    city_encoded = le_city.transform([city])[0]

    car_age = 2026 - year

    input_data = pd.DataFrame({
        "Make": [make_encoded],
        "Model": [model_encoded],
        "Year": [year],
        "KM's driven": [kms],
        "Fuel": [fuel_encoded],
        "Registration city": [city_encoded],
        "Car_Age": [car_age]
    })

    prediction = model.predict(input_data)[0]

    prediction = np.expm1(prediction)

    st.write("")

    st.markdown(
        f'<p class="prediction">Estimated Price: PKR {prediction:,.0f}</p>',
        unsafe_allow_html=True
    )

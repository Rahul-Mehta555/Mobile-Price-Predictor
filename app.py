import joblib
import numpy as np
import pandas as pd
import streamlit as st
DATA_PATH = "Cellphone.xlsx"
MODEL_PATH = "Mobile Price Predictor.pk1"
LOG_COLUMNS = ["ram", "thickness", "cpu freq", "battery", "weight"]


def load_data():
    data = pd.read_excel(DATA_PATH)
    data = data.drop(columns=["Product_id"], errors="ignore")

    if "resoloution" in data.columns:
        data = data.rename(columns={"resoloution": "resolution"})

    return data.drop_duplicates()


def load_model():
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Mobile Price Predictor", page_icon="📱")
st.title("📱 Mobile Price Predictor")
st.write("Enter the phone specifications to estimate its price.")

try:
    data = load_data()
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the dataset or model: {exc}")
    st.stop()

if "Price" not in data.columns:
    st.error("The dataset must contain a 'Price' column.")
    st.stop()

features = data.drop(columns="Price")
model_features = list(model.named_steps["scaler"].feature_names_in_)
if set(features.columns) != set(model_features):
    st.error("The dataset features do not match the features expected by the model.")
    st.stop()
features = features.loc[:, model_features]
if not all(pd.api.types.is_numeric_dtype(features[col]) for col in features.columns):
    st.error("The model expects numeric features, but the dataset contains non-numeric columns.")
    st.stop()

with st.form("phone_specs"):
    values = {}

    for column in features.columns:
        series = features[column].dropna()
        if series.empty:
            st.error(f"No usable values found for {column}.")
            st.stop()

        low = float(series.min())
        high = float(series.max())
        default = float(series.median())

        # Ensure the input bounds are valid.
        if low == high:
            low -= 1
            high += 1

        values[column] = st.number_input(
            column,
            min_value=low,
            max_value=high,
            value=min(max(default, low), high),
        )

    submitted = st.form_submit_button("Predict price")

if submitted:
    input_df = pd.DataFrame([values], columns=features.columns)

    for column in LOG_COLUMNS:
        if column in input_df.columns:
            input_df[column] = np.log1p(input_df[column])

    try:
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted price: {prediction:.2f}")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
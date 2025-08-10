import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# --- Custom feature engineering function ---
# IMPORTANT: Replace with the exact same logic you used when training
def add_engineered_features(df):
    """
    Adds engineered features to the dataframe.
    Must match the logic used during training to ensure consistency.
    """
    if "numberOfRooms" in df.columns and "floors" in df.columns:
        df["rooms_per_floor"] = df["numberOfRooms"] / df["floors"].replace(0, np.nan)
        df["rooms_per_floor"].fillna(0, inplace=True)
    return df

# --- Page Configuration ---
st.set_page_config(
    page_title="🏠 Paris Housing Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 Paris Housing Price Predictor")

# --- Load files directly from repository ---
MODEL_FILE = "price_model.pkl"
FEATURES_FILE = "feature_cols.pkl"
PREPROCESSOR_FILE = "preprocessor.pkl"

@st.cache_resource
def load_files():
    """
    Loads the model, feature list, and preprocessor directly from the project directory.
    """
    model = joblib.load(MODEL_FILE)
    feature_cols = joblib.load(FEATURES_FILE)
    preprocessor = joblib.load(PREPROCESSOR_FILE)
    return model, feature_cols, preprocessor

try:
    model, feature_cols, preprocessor = load_files()
except Exception as e:
    st.error(f"Failed to load model files. Ensure they are present in the repository.\nError: {e}")
    st.stop()

# --- User Input Form ---
st.markdown("Enter the property features below and click **Predict Price**.")

defaults = {
    "squareMeters": 100, "numberOfRooms": 3, "hasYard": 1, "hasPool": 0, "floors": 2,
    "cityCode": 75001, "cityPartRange": 5, "numPrevOwners": 1, "made": 2010,
    "isNewBuilt": 1, "hasStormProtector": 1, "basement": 1, "attic": 1,
    "garage": 1, "hasStorageRoom": 1, "hasGuestRoom": 1
}

with st.form("price_form"):
    c1, c2 = st.columns(2)
    user_input = {}
    with c1:
        user_input["squareMeters"] = st.number_input("Square Meters", min_value=1, value=defaults["squareMeters"])
        user_input["numberOfRooms"] = st.number_input("Number of Rooms", min_value=0, value=defaults["numberOfRooms"])
        user_input["floors"] = st.number_input("Floors", min_value=0, value=defaults["floors"])
        user_input["made"] = st.number_input("Year Built (e.g., 2010)", min_value=1800, max_value=2100, value=defaults["made"])
        user_input["cityCode"] = st.number_input("City Code", min_value=1, value=defaults["cityCode"])
        user_input["cityPartRange"] = st.number_input("City Part Range", min_value=1, max_value=10, value=defaults["cityPartRange"])
        user_input["numPrevOwners"] = st.number_input("Previous Owners", min_value=0, value=defaults["numPrevOwners"])
        user_input["hasGuestRoom"] = st.number_input("Guest Rooms (count or 0/1)", min_value=0, value=defaults["hasGuestRoom"])
    with c2:
        user_input["hasYard"] = st.selectbox("Has Yard?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=defaults["hasYard"])
        user_input["hasPool"] = st.selectbox("Has Pool?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=defaults["hasPool"])
        user_input["isNewBuilt"] = st.selectbox("Is New Built?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=defaults["isNewBuilt"])
        user_input["hasStormProtector"] = st.selectbox("Has Storm Protector?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=defaults["hasStormProtector"])
        user_input["hasStorageRoom"] = st.selectbox("Has Storage Room?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=defaults["hasStorageRoom"])
        user_input["basement"] = st.number_input("Basement Size/Indicator", min_value=0, value=defaults["basement"])
        user_input["attic"] = st.number_input("Attic Size/Indicator", min_value=0, value=defaults["attic"])
        user_input["garage"] = st.number_input("Garage Size/Indicator", min_value=0, value=defaults["garage"])

    submit = st.form_submit_button("Predict Price")

# --- Prediction Logic ---
if submit:
    input_df = pd.DataFrame([user_input])

    try:
        processed_df = preprocessor(input_df)
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        st.stop()

    for col in feature_cols:
        if col not in processed_df.columns:
            processed_df[col] = 0
    aligned_df = processed_df[feature_cols]

    try:
        prediction = model.predict(aligned_df)
        st.success(f"Predicted Price: €{float(prediction[0]):,.2f}")
    except Exception as e:
        st.error(f"Prediction failed. Please check the input values.\nError: {e}")

# --- Diagnostics Expander ---
with st.expander("Diagnostics"):
    st.write("Model File Found:", Path(MODEL_FILE).exists())
    st.write("Feature List File Found:", Path(FEATURES_FILE).exists())
    st.write("Preprocessor File Found:", Path(PREPROCESSOR_FILE).exists())
    if 'aligned_df' in locals():
        st.write("Input DataFrame sent to model:")
        st.dataframe(aligned_df)
    st.write("Features expected by the model:", feature_cols)

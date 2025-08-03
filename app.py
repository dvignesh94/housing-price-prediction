import streamlit as st
import pandas as pd
import joblib

# Loading the model and scaler
model = joblib.load("/Users/vignesh/Downloads/Paris Housing Price Prediction/price_model.pkl")
scaler = joblib.load("/Users/vignesh/Downloads/Paris Housing Price Prediction/scaler.pkl")

st.title("🏠 Paris Housing Price Predictor")

# Input fields with default values
fields = {
    "squareMeters": 100, "numberOfRooms": 3, "hasYard": 1, "hasPool": 0, "floors": 2,
    "cityCode": 75001, "cityPartRange": 5, "numPrevOwners": 1, "made": 2010,
    "isNewBuilt": 1, "hasStormProtector": 1, "basement": 1, "attic": 1,
    "garage": 1, "hasStorageRoom": 1, "hasGuestRoom": 1
}

user_input = {}
for field, default in fields.items():
    user_input[field] = st.number_input(f"{field}", value=default)

# Predict button
if st.button("Predict Price"):
    # Converting the user input to DataFrame
    input_df = pd.DataFrame([user_input])

    # Calculate derived feature: rooms_per_floor
    input_df["rooms_per_floor"] = input_df["numberOfRooms"] / input_df["floors"]
    input_df["rooms_per_floor"] = input_df["rooms_per_floor"].replace([float("inf"), -float("inf")], 0).fillna(0)

    # Scaling input and predict
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    # Displaying the result
    st.success(f"Predicted Price: €{prediction[0]:,.2f}")

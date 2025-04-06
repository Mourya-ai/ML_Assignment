import streamlit as st
import numpy as np
from keras.models import load_model
from keras.losses import mean_squared_error as mse

# Set up the title and description
st.title("💹 INR Price Prediction App")
st.markdown("""
Predict the next **USD to INR** exchange closing price based on recent trends (2019–2023).

> Enter the last 5 closing prices below, and our LSTM model will predict the next price.
""")

# User inputs for the last 5 closing prices
st.subheader("Enter Last 5 Closing Prices (USD to INR)")
col1, col2, col3, col4, col5 = st.columns(5)
with col1: price1 = st.number_input("Price 1", value=86.23)
with col2: price2 = st.number_input("Price 2", value=86.54)
with col3: price3 = st.number_input("Price 3", value=86.90)
with col4: price4 = st.number_input("Price 4", value=87.20)
with col5: price5 = st.number_input("Price 5", value=86.63)

# Prediction section
if st.button("Predict Next Closing Price"):
    try:
        # Load the LSTM model
        model = load_model("lstm_model.h5", custom_objects={"mse": mse})

        # Prepare the input
        input_data = np.array([[price1, price2, price3, price4, price5]])
        input_data = input_data.reshape((input_data.shape[0], 1, input_data.shape[1]))

        # Make prediction
        prediction = model.predict(input_data)

        # Show result
        predicted_price = prediction[0][0]
        st.success(f"📈 Predicted Next Closing Price: **₹{predicted_price:.2f}**")

    except Exception as e:
        st.error(f"Error: {e}")
        st.warning("Ensure `lstm_model.h5` is in the same directory as this app.")


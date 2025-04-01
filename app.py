import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import snowflake.connector
from dotenv import load_dotenv
import os

# Load models
with open("XG_model.pkl", 'rb') as file:
    xg_model = pickle.load(file)

with open("KNN_model.pkl", 'rb') as file:
    knn_model = pickle.load(file)

models = {
    "XG Regressor": xg_model,
    "KNN Regressor": knn_model,
}

st.title("Amazon Stock Volume Prediction (2000-2025)")
st.markdown("By ABHISHEK")
st.write("Enter the data to get the prediction")
name = st.text_input("Name")
email = st.text_input("Email")

st.write("Choose the model to predict the stock volume")
selected_model = st.selectbox("Choose a model", list(models.keys()))

open = st.slider("Opening Price ($)", min_value=0.0, max_value=250.0, step=1.0, value=100.0)
high = st.slider("Highest Price ($)", min_value=0.0, max_value=250.0, step=1.0, value=100.0)
low = st.slider("Lowest Price ($)", min_value=0.0, max_value=250.0, step=1.0, value=100.0)
close = st.slider("Closing Price ($)", min_value=0.0, max_value=250.0, step=1.0, value=100.0)
adj_close = st.slider("Adjusted Closing Price ($)", min_value=0.0, max_value=250.0, step=1.0, value=100.0)

# Initialize lists to store inputs and predictions
if 'inputs' not in st.session_state:
    st.session_state.inputs = []
if 'predictions' not in st.session_state:
    st.session_state.predictions = []

prediction = None
if st.button("Predict"):
    if not name or not email:
        st.error("Please enter your name and email.")
    else:
        model = models[selected_model]
        user_input = [float(open), float(high), float(low), float(close), float(adj_close)]
        prediction = float(model.predict([user_input])[0])
        
        # Store the input and prediction
        st.session_state.inputs.append(user_input)
        st.session_state.predictions.append(prediction)
        
        if len(st.session_state.inputs) > 5:
            st.session_state.inputs.pop(0)
            st.session_state.predictions.pop(0)

        # cursor = conn.cursor()
        # insert_query = """
        # INSERT INTO app_data (name, email, selected_model, open, high, low, close, adj_close, prediction)
        # VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        # """
        # cursor.execute(insert_query, (name, email, selected_model, float(open), float(high), float(low), float(close), float(adj_close), float(prediction)))
        # cursor.close()

if prediction is not None:
    st.success(f"The model predicts: {prediction} stock in volume")

    stored_data = pd.DataFrame(st.session_state.inputs, columns=['Open', 'High', 'Low', 'Close', 'Adj Close'])
    stored_data['Prediction'] = st.session_state.predictions

    fig, ax = plt.subplots()
    ax.plot(stored_data.index, stored_data['Prediction'], marker='x', label='Predicted Volume')
    ax.set_xlabel('Prediction Index')
    ax.set_ylabel('Predicted Volume')
    ax.set_title('Stored Predictions')
    ax.legend()

    st.pyplot(fig)

# conn.close()
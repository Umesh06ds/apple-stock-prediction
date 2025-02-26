
import streamlit as st  
import pickle  
import pandas as pd  

# Load trained models  
mlr_model = pickle.load(open("mlr_model.pkl", "rb"))  
rf_model = pickle.load(open("rf_model.pkl", "rb"))  
xgb_model = pickle.load(open("xgb_model.pkl", "rb"))  

# Streamlit UI  
st.title("📈 Apple Stock Price Prediction App")  
st.write("Enter stock price details below and get predictions.")  

# Input fields  
open_price = st.number_input("Open Price", min_value=0.0, step=0.1, format="%.2f")  
high_price = st.number_input("High Price", min_value=0.0, step=0.1, format="%.2f")  
low_price = st.number_input("Low Price", min_value=0.0, step=0.1, format="%.2f")  
close_price = st.number_input("Close Price", min_value=0.0, step=0.1, format="%.2f")  

# Predict button  
if st.button("Predict"):  
    # Prepare input data  
    data = pd.DataFrame([[open_price, high_price, low_price, close_price]],  
                        columns=['Open', 'High', 'Low', 'Close'])  

    # Make predictions  
    mlr_pred = mlr_model.predict(data)[0]  
    rf_pred = rf_model.predict(data)[0]  
    xgb_pred = xgb_model.predict(data)[0]  

    # Display results  
    st.subheader("📊 Predictions:")  
    st.write(f"🔹 **MLR Prediction:** {mlr_pred}")  
    st.write(f"🔹 **Random Forest Prediction:** {rf_pred}")  
    st.write(f"🔹 **XGBoost Prediction:** {xgb_pred}")  

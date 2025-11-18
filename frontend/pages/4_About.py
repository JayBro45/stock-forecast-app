import streamlit as st
from components.navbar import navbar

navbar("About")

st.title("ℹ️ About the Project")
st.write("""
This application was built using:

- **FastAPI** backend  
- **Streamlit** frontend  
- XGBoost, LSTM, ARIMA, NeuralProphet  
- Live forecasting with 30-day horizon
""")

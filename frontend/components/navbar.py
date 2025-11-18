import streamlit as st

def navbar(title="Stock Forecast App"):
    st.markdown(
        f"""
        <div class="navbar">
            <h2 style="color:white; margin:0;">{title}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

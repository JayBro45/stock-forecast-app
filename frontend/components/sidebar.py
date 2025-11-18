import streamlit as st
from PIL import Image

def load_sidebar():
    logo = Image.open("frontend/assets/logo.png")
    st.sidebar.image(logo, width=160)
    st.sidebar.markdown("---")

    st.sidebar.header("Navigation")
    st.sidebar.write("Use the main sidebar controls to switch pages.")

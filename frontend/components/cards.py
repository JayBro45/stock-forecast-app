import streamlit as st

def metric_card(title, value, color="#4CAF50"):
    st.markdown(
        f"""
        <div class="card">
            <h4 style="margin-bottom:4px;">{title}</h4>
            <h2 style="color:{color};">{value}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )


import sklearn
import streamlit as st
from predictor import show_predict_page

page = st.sidebar.selectbox("Predict", ("Predict", ))

if page == "Predict":
    show_predict_page()
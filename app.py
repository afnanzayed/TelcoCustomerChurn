import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Title
st.title("Data Science Project - Streamlit App")

# Upload Data
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(data.head())

    # Analysis options
    if st.checkbox("Show Summary Statistics"):
        st.write(data.describe())

    # Visualization
    if st.checkbox("Visualize Data"):
        column = st.selectbox("Choose a column to visualize:", data.columns)
        plt.hist(data[column].dropna(), bins=20)
        st.pyplot(plt)

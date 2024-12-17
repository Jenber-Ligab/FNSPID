import streamlit as st
import matplotlib.pyplot as plt
from stock_analysis import load_data, plot_stock_data, plot_rsi, plot_macd
from fetch_stock_data import fetch_historical_data
# Streamlit app starts here
st.title("Stock Analysis Dashboard")

# File upload
uploaded_file = st.file_uploader("Upload Stock Data CSV", type="csv")
if uploaded_file:
    # Load data
    df = load_data(uploaded_file)
    st.write("Data Preview:", df.head())

    # Sidebar: Stock selection
    stocks = df['stock'].unique()
    selected_stock = st.sidebar.selectbox("Select a Stock", stocks)

    # Stock price analysis
    st.subheader("Stock Price Analysis")
    fig = plot_stock_data(selected_stock, df)
    st.pyplot(fig)

    # RSI analysis
    st.subheader("Relative Strength Index (RSI)")
    fig = plot_rsi(selected_stock, df)
    st.pyplot(fig)

    # MACD analysis
    st.subheader("Moving Average Convergence Divergence (MACD)")
    fig = plot_macd(selected_stock, df)
    st.pyplot(fig)
else:
    st.info("Please upload a stock data CSV file to begin analysis.")
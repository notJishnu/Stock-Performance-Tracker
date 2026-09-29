
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import date, timedelta

# ---------------------------------
# 1. Page Configuration & UI Header
# ---------------------------------
st.set_page_config(page_title="Market Tracker", layout="wide")
st.title("Interactive Stock Performance Tracker")
st.markdown("Compare the relative growth of multiple assets over time. All selected assets are normalized to a base value of 100 at the start date for an exact percentage-growth comparison.")

# ---------------------------------
# 2. Sidebar & User Inputs
# ---------------------------------
st.sidebar.header("Filter Options")

available_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "SPY", "QQQ"]
selected_tickers = st.sidebar.multiselect(
    "Select Tickers (or type your own):", 
    available_tickers,
    default=["AAPL", "MSFT", "SPY"]
)

default_start = date.today() - timedelta(days=365)
start_date = st.sidebar.date_input("Start Date", default_start)
end_date = st.sidebar.date_input("End Date", date.today())

# ---------------------------------
# 3. Data Ingestion & Caching
# ---------------------------------
@st.cache_data
def load_data(tickers, start, end):
    if not tickers:
        return pd.DataFrame()
    
    # Notice we are pulling 'Close' instead of 'Adj Close' to fix the yfinance update
    df = yf.download(tickers, start=start, end=end)['Close']
    
    if len(tickers) == 1:
        df = pd.DataFrame(df)
        df.columns = tickers
        
    return df

# ---------------------------------
# 4. Data Processing & Visualization
# ---------------------------------
if not selected_tickers:
    st.warning("Please select at least one ticker from the sidebar to generate the dashboard.")
else:
    raw_data = load_data(selected_tickers, start_date, end_date)
    
    if raw_data.empty:
        st.error("No data found for the selected date range or tickers.")
    else:
        clean_data = raw_data.ffill()
        normalized_data = (clean_data / clean_data.iloc[0]) * 100
        
        plot_data = normalized_data.reset_index()
        melted_data = plot_data.melt(
            id_vars=['Date'], 
            var_name='Ticker', 
            value_name='Normalized Price'
        )
        
        fig = px.line(
            melted_data, 
            x='Date', 
            y='Normalized Price', 
            color='Ticker',
            title=f"Relative Performance ({start_date} to {end_date})"
        )
        
        fig.update_layout(
            hovermode="x unified",
            yaxis_title="Normalized Value (Base 100)",
            xaxis_title="",
            legend_title="Assets"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Raw Normalized Data")
        st.dataframe(normalized_data.tail().sort_index(ascending=False), use_container_width=True)
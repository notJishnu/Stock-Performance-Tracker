
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import date, timedelta
from prophet import Prophet
import plotly.graph_objects as go

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

        # ---------------------------------
        # 5. Time-Series Forecasting
        # ---------------------------------
        st.divider()
        st.subheader("Predictive Forecasting (Meta Prophet)")
        st.markdown("Forecast future price action based on historical trends. The shaded region represents the model's confidence interval.")
        
        # User controls for the forecast
        col1, col2 = st.columns(2)
        with col1:
            forecast_ticker = st.selectbox("Select Asset to Forecast:", selected_tickers)
        with col2:
            forecast_days = st.slider("Days into the future:", 30, 365, 90)

        # We use a button so the model doesn't re-train every time the user clicks a filter
        if st.button("Generate Forecast"):
            with st.spinner(f"Training Prophet model for {forecast_ticker}..."):
                # 1. Prepare Data for Prophet (requires 'ds' and 'y' columns)
                # We use raw_data here to predict actual prices, not base-100 normalized data
                df_prophet = raw_data[[forecast_ticker]].copy().reset_index()
                df_prophet.columns = ['ds', 'y']
                
                # Drop any remaining NaNs to prevent model failure
                df_prophet = df_prophet.dropna()

                # 2. Initialize and Train the Model
                model = Prophet(daily_seasonality=True)
                model.fit(df_prophet)

                # 3. Create Future Dates and Predict
                future_dates = model.make_future_dataframe(periods=forecast_days)
                forecast = model.predict(future_dates)

                # 4. Visualize with Plotly
                # 4. Visualize with Custom Plotly Graph Objects
                fig_forecast = go.Figure()

                # Add Historical Data
                fig_forecast.add_trace(go.Scatter(
                    x=df_prophet['ds'], y=df_prophet['y'], 
                    name="Historical", mode="lines", line=dict(color="#1f77b4")
                ))

                # Add Forecast Line
                fig_forecast.add_trace(go.Scatter(
                    x=forecast['ds'], y=forecast['yhat'], 
                    name="Prediction", mode="lines", line=dict(color="#ff7f0e")
                ))

                # Add Confidence Interval (Upper bound - invisible line)
                fig_forecast.add_trace(go.Scatter(
                    x=forecast['ds'], y=forecast['yhat_upper'],
                    mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"
                ))
                
                # Add Confidence Interval (Lower bound - fills space to the upper bound)
                fig_forecast.add_trace(go.Scatter(
                    x=forecast['ds'], y=forecast['yhat_lower'],
                    mode="lines", line=dict(width=0), fill="tonexty", 
                    fillcolor="rgba(255, 127, 14, 0.2)", name="Confidence Interval", hoverinfo="skip"
                ))

                fig_forecast.update_layout(
                    title=f"{forecast_ticker} Price Forecast ({forecast_days} Days)",
                    yaxis_title="Asset Price",
                    xaxis_title="",
                    hovermode="x unified"
                )
                
                
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                # Show the raw prediction bounds for transparency
                st.write("Forecast Data (Upper & Lower Bounds):")
                st.dataframe(
                    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days),
                    use_container_width=True
                )
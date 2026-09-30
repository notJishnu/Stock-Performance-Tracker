# 📈 Interactive Stock Performance & Forecasting Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_LIVE_APP_URL_HERE)

## Overview
An end-to-end financial data analytics pipeline and interactive dashboard designed to extract, transform, and visualize live equity market data. This application enables accurate comparative analysis by normalizing asset growth and leverages machine learning to forecast future price action with statistical confidence intervals. 

Built to demonstrate production-level data engineering, API integration, and predictive modeling within a unified, deployed web interface.

## Key Features
* **Dynamic Base-100 Normalization:** Standardizes multi-asset performance by reindexing disparate stock prices to a base value of 100. This allows for exact percentage-growth comparisons across custom time horizons regardless of the initial share price.
* **Predictive Forecasting (Meta Prophet):** Integrates Meta's open-source Prophet library to train models on the fly and generate 30-to-365 day time-series price predictions that account for daily market seasonality.
* **Custom Statistical Visualizations:** Bypasses standard plotting limitations by utilizing `plotly.graph_objects` to render interactive charts featuring custom shaded confidence intervals (upper and lower probability bounds) for machine learning forecasts.
* **Automated Data Pipeline:** Programmatically ingests real-time and historical pricing data via the Yahoo Finance API, cleaning missing data points and utilizing in-memory caching to optimize speed and prevent API rate-limiting.

## Technology Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Web Framework** | Streamlit | Cloud deployment, UI architecture, state management |
| **Data Engineering** | Pandas, yfinance | REST API extraction, time-series indexing, forward-filling |
| **Machine Learning** | Prophet | Time-series prediction modeling |
| **Visualization** | Plotly (Graph Objects) | Interactive, multi-layered visual rendering |

## How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_GITHUB_USERNAME/stock-performance-tracker.git](https://github.com/YOUR_GITHUB_USERNAME/stock-performance-tracker.git)
   cd stock-performance-tracker
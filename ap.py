import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go

# Page Config
st.set_page_config(layout="wide", page_title="Stock Market Prediction", page_icon="📈")

st.image("https://i.postimg.cc/j2tdGvv5/new-removebg-preview.png", width=200)

# Custom CSS for Translucent Sidebar and Responsiveness
responsive_css = """
<style>
    /* Sidebar adjustments */
    [data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.1);
        max-width: 100%;
    }
    /* General adjustments for smaller screens */
    @media screen and (max-width: 768px) {
        .stApp {
            padding: 10px;
        }
        header[data-testid="stHeader"] {
            text-align: center;
        }
        .css-1aumxhk, .css-1629p8f {
            font-size: 1.5rem;
        }
    }
    /* Footer for small screens */
    @media screen and (max-width: 768px) {
        .footer {
            font-size: 0.8rem;
        }
    }
</style>
"""
st.markdown(responsive_css, unsafe_allow_html=True)

# Background Image
background_css = """
<style>
    .stApp {
        background-image: url('https://wallpapercg.com/download/candlestick-pattern-7680x4320-19473.jpg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
</style>
"""
st.markdown(background_css, unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.header("📊 Stock Market Prediction")

# Input Section for Ticker (Dropdown and Custom Input)
st.sidebar.markdown(
    """
    <h2 style='font-size: 24px; color: #ff6347; text-align: center;'>Select or Enter a Stock Ticker</h2>
    """, unsafe_allow_html=True)

# Dropdown menu for predefined tickers
stock_tickers = [
    "AAPL", "TSLA", "GOOGL", "AMZN", "MSFT", "HDB", "BOAT", "ZOMATO.BO", 
    "BAJAJFINSV.NS", "^NSEI", "^BSESN", "INFY.NS", "ADANIGREEN.NS", "ADANIENT.NS", 
    "RELIANCE.NS", "ITC.NS", "HINDUNILVR.BO", "TCS.NS", "TATASTEEL.NS", "TATAPOWER.BO",
    "ADANIPOWER.NS", "AXISBANK.NS", "ASIANPAINT.BO", "ADANIPORTS.BO", "BAJAJFINSV.NS", 
    "BAJAJ-AUTO.BO", "BHARTIARTL.BO", "CIPLA.BO", "ICICIBANK.NS", "INDUSINDBK.BO", 
    "JSWSTEEL.NS", "KOTAKBANK.NS", "MARUTI.BO", "ONGC.BO", "SPARC.NS", "TATAMOTORS.BO", 
    "TECHM.BO", "WIPRO.NS", "BANKBARODA.BO", "IRCTC.BO", "IDFCFIRSTB.NS", "DABUR.BO", 
    "COALINDIA.BO", "CANBK.NS", "JINDALSTEL.BO", "BANDHANBNK.BO", "ULTRACEMCO.BO", 
    "DELHIVERY.NS", "KALYANKJIL.BO", "ZYDUSLIFE.BO"
]

# Dropdown for predefined tickers
selected_ticker = st.sidebar.selectbox("Choose a Ticker", stock_tickers)

# Text input for custom ticker (with large font size and color)
custom_ticker = st.sidebar.text_input("Or enter a custom ticker:", value="", help="Enter the stock ticker symbol from Yahoo Finance")

# Display both options
ticker_to_use = custom_ticker if custom_ticker else selected_ticker

# Display chosen ticker with big font and color
st.sidebar.markdown(f"""
    <h2 style='font-size: 28px; color: #ff6347;'>Using Ticker: <b>{ticker_to_use}</b></h2>
""", unsafe_allow_html=True)

# Fetch Stock Data
data = yf.download(ticker_to_use, period="1y")

if data.empty:
    st.error("No data available for this ticker. Please try another.")
else:
    # Select relevant columns
    data = data[['Open', 'Close', 'High', 'Low']]

    # Title and Description
    st.title(f"{ticker_to_use.upper()} Stock Price Prediction")
    st.subheader("Historical Stock Data")
    st.write("The table below shows historical data for the selected stock.")

    # Display raw data
    st.dataframe(
        data.style.set_properties(**{'background-color': '#1e1e1e', 'color': '#ffffff'}),
        height=400,
        use_container_width=True
    )

    # Feature Engineering
    data['Target'] = data['Close'].shift(-1)
    data.dropna(inplace=True)

    if len(data) < 2:
        st.error("Not enough data after processing for training. Please try a different ticker.")
    else:
        # Split data into features and target
        X = data[['Open', 'High', 'Low', 'Close']]
        y = data['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model Training
        model = XGBRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Model Accuracy
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Forecast for 10 days
        future_days = 10
        last_data = X.iloc[-1].values.reshape(1, -1)
        future_predictions = []
        for _ in range(future_days):
            pred_price = model.predict(last_data)
            future_predictions.append(pred_price[0])
            last_data = np.append(last_data[:, 1:], pred_price).reshape(1, -1)

        # Model Performance Metrics
        st.subheader("📈 Model Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean Absolute Error", f"{mae:.2f}")
        col2.metric("Root Mean Squared Error", f"{rmse:.2f}")
        col3.metric("R² Score", f"{r2:.2f}")

        # Historical and Predicted Data Visualization
        st.subheader("📉 Actual vs Predicted Prices")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=data.index[-len(y_test):], y=y_test, mode='lines', name='True Closing Price', line=dict(color='#00ff7f')))
        fig1.add_trace(go.Scatter(x=data.index[-len(y_test):], y=y_pred, mode='lines', name='Predicted Closing Price', line=dict(color='#ff4500')))
        fig1.update_layout(template='plotly_dark', title="True vs Predicted Closing Prices", xaxis_title="Date", yaxis_title="Price", legend_title="Legend")
        st.plotly_chart(fig1, use_container_width=True)

        
# Forecast Visualization
st.subheader("🔮 Forecasted Prices (Next 10 Business Days)")
# Ensure the next business day is correctly computed
next_business_day = pd.Timestamp.today() + pd.offsets.BDay(1)  # Get the next business day (skipping weekends)
future_dates = pd.date_range(start=next_business_day, periods=future_days, freq='B')  # Use freq='B' to skip weekends

# Format the dates to only show the date (no time)
future_dates = future_dates.date  # This will remove the time part

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=future_dates, y=future_predictions, mode='lines+markers', name="Forecasted Prices", line=dict(color='#00ff7f')))
fig2.update_layout(template='plotly_dark', title="Forecast for Next 10 Business Days", xaxis_title="Date", yaxis_title="Price", legend_title="Legend")
st.plotly_chart(fig2, use_container_width=True)

# Forecast Table
st.subheader("📋 Forecasted Prices Table")
forecast_df = pd.DataFrame({"Date": future_dates, "Forecasted Price": future_predictions})

# Display the table without the index
st.table(forecast_df.style.set_properties(**{'background-color': '#1e1e1e', 'color': '#ffffff'}).hide(axis='index'))







# Footer
st.markdown("---")
st.markdown(
    """
    <div class="footer" style="text-align: center;">
        <p style="font-size: 1.2rem; color: #ff6347;">© 2025 SPRED Stock Market Prediction App. All rights reserved.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


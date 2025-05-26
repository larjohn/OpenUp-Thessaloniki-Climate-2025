import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import datetime

# Title
st.title("🌫️ Unified AQI Forecast & Daily Estimation")

# Sidebar Controls
st.sidebar.title("🔧 Settings")
forecast_days = st.sidebar.slider("Forecast period (days)", min_value=7, max_value=365, value=30)

enable_manual = st.sidebar.checkbox("Estimate Tomorrow's AQI from Today's Pollutant Data")

# Default Placeholder Values for Pollutants (you can adjust these as needed)
placeholder_values = {
    "pm10": 30.0,  # Example: Moderate level
    "pm2.5": 20.0,  # Example: Moderate level
    "co": 0.5,      # Example: CO level in mg/m³
    "no": 20.0,     # Example: Nitrogen monoxide level in µg/m³
    "no2": 40.0,    # Example: Nitrogen dioxide level in µg/m³
    "so2": 10.0,    # Example: Sulfur dioxide level in µg/m³
    "o3": 50.0,     # Example: Ozone level in µg/m³
}

# Manual Input Section
manual_input = {}
selected_date = datetime.date.today()
if enable_manual:
    st.sidebar.markdown("### 🧪 Pollutant Data at 12:00 PM")
    selected_date = st.sidebar.date_input("Date of measurement", value=datetime.date.today())
    manual_input = {
        "pm10": st.sidebar.number_input("PM10 (µg/m³)", min_value=0.0, step=1.0, value=placeholder_values["pm10"]),
        "pm2.5": st.sidebar.number_input("PM2.5 (µg/m³)", min_value=0.0, step=1.0, value=placeholder_values["pm2.5"]),
        "co": st.sidebar.number_input("CO (mg/m³)", min_value=0.0, step=0.1, value=placeholder_values["co"]),
        "no": st.sidebar.number_input("NO (µg/m³)", min_value=0.0, step=1.0, value=placeholder_values["no"]),
        "no2": st.sidebar.number_input("NO2 (µg/m³)", min_value=0.0, step=1.0, value=placeholder_values["no2"]),
        "so2": st.sidebar.number_input("SO2 (µg/m³)", min_value=0.0, step=1.0, value=placeholder_values["so2"]),
        "o3": st.sidebar.number_input("O3 (µg/m³)", min_value=0.0, step=1.0, value=placeholder_values["o3"]),
    }

# --- Load Data ---
DATA_URL = "https://raw.githubusercontent.com/PanosKats/OpenUp-Thessaloniki-Climate-2025/refs/heads/master/Data/Final_Data.csv"  # Replace with your raw CSV URL

@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    df['time'] = pd.to_datetime(df['time'], dayfirst=True)
    df = df.rename(columns={'time': 'ds'})
    return df

try:
    # Load and prepare data
    df = load_data(DATA_URL)
    df_multi = df[['ds', 'aqi', 'pm10', 'pm2.5', 'co', 'no', 'no2', 'so2', 'o3']].dropna()
    df_multi = df_multi.rename(columns={'aqi': 'y'})

    st.subheader("📄 Raw AQI Data")
    st.write(df_multi[['ds', 'y']].tail())

    st.subheader("📉 AQI Time Series Overview")
    st.line_chart(df_multi.set_index('ds')['y'])

    # --- Unified Prophet Model ---
    st.subheader(f"🔮 Unified Forecast for Next {forecast_days} Days")

    model = Prophet()
    for reg in ['pm10', 'pm2.5', 'co', 'no', 'no2', 'so2', 'o3']:
        model.add_regressor(reg)
    model.fit(df_multi)

    # Prepare future for multi-day forecast
    last_date = df_multi['ds'].max()
    future_days = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
    recent_mean = df_multi[['pm10', 'pm2.5', 'co', 'no', 'no2', 'so2', 'o3']].mean()

    future_df = pd.DataFrame({'ds': future_days})
    for col in recent_mean.index:
        future_df[col] = recent_mean[col]

    full_future = pd.concat([df_multi[['ds'] + list(recent_mean.index)], future_df], ignore_index=True)
    forecast = model.predict(full_future)

    st.subheader("📈 AQI Forecast")
    st.plotly_chart(plot_plotly(model, forecast))

    # --- Custom Hourly Prediction for Tomorrow ---
    if enable_manual:
        st.subheader("🕛 Forecast Tomorrow's AQI from 12:00 PM to Midnight")

        tomorrow_date = selected_date + datetime.timedelta(days=1)
        noon_time = datetime.datetime.combine(tomorrow_date, datetime.time(hour=12))
        hourly_range = pd.date_range(start=noon_time, end=noon_time.replace(hour=23, minute=59), freq='H')

        # Prepare the dataframe for tomorrow's prediction
        manual_df = pd.DataFrame({'ds': hourly_range})
        for col in manual_input:
            manual_df[col] = manual_input[col]

        forecast_manual = model.predict(manual_df)

        # --- Display Hourly AQI Forecast Table ---
        st.subheader("📋 Hourly AQI Forecast Table for Tomorrow")
        forecast_table = forecast_manual[['ds', 'yhat']].rename(columns={'ds': 'Time', 'yhat': 'Predicted AQI'})
        st.dataframe(forecast_table)

        # --- Plot Simple Time Series Graph ---
        st.subheader("📊 Simple Time Series of Hourly AQI Forecast")
        st.line_chart(forecast_table.set_index('Time')['Predicted AQI'])

        # Display the final AQI value at midnight
        end_aqi = forecast_manual['yhat'].iloc[-1]
        st.success(f"🎯 Estimated AQI at Midnight Tomorrow: **{end_aqi:.2f}**")

except Exception as e:
    st.error(f"❌ Failed to load data or forecast: {e}")

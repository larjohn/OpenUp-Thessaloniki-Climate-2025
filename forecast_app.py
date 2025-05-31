import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import datetime
import plotly.express as px
import plotly.graph_objects as go




DATA_URL = "https://raw.githubusercontent.com/PanosKats/OpenUp-Thessaloniki-Climate-2025/master/Data/Final_Data.csv"
WEATHER_DATA_URL = "https://raw.githubusercontent.com/PanosKats/OpenUp-Thessaloniki-Climate-2025/master/Data/Weather_1_1_22_30_06_24.csv"




# AQI Data
@st.cache_data
def load_aqi_data():
    df = pd.read_csv(DATA_URL)
    df['time'] = pd.to_datetime(df['time'])
    df = df.rename(columns={'time': 'ds'})
    return df

# Weather Data 
@st.cache_data
def load_weather_data():
    df = pd.read_csv(WEATHER_DATA_URL)
    df = df[df["time"] != "time"]  # Remove any redundant header rows
    df.columns = ['time', 'temperature_2m', 'relative_humidity_2m', 'weather_code', 'precipitation']
    df['ds'] = pd.to_datetime(df['time'], errors='coerce')
    df.dropna(subset=['ds'], inplace=True)
    return df


# Time-Lagged Correlation
def compute_time_lagged_correlation(df, col, lags):
    correlations = []
    for lag in lags:
        shifted = df[col].shift(lag)
        corr = df['aqi'].corr(shifted)
        correlations.append(corr)
    return correlations


st.set_page_config(page_title="AQI Forecast App", layout="wide")


st.sidebar.title("Aqi Prediction")
forecast_days = st.sidebar.slider("Forecast period (days)", min_value=7, max_value=365, value=30)
enable_manual = st.sidebar.checkbox("Estimate AQI from today's pollutant data")



# INPUT 
manual_input = {}
selected_date = datetime.date.today()
if enable_manual:
    st.sidebar.markdown("### Pollutant Data at 12:00 PM")
    selected_date = st.sidebar.date_input("Date of measurement", value=datetime.date.today())
    default_values = load_aqi_data()[['pm10', 'pm2.5', 'co', 'no', 'no2', 'so2', 'o3']].mean()
    manual_input = {
        key: st.sidebar.number_input(key.upper(), min_value=0.0, step=1.0 if key != 'co' else 0.1, value=float(f"{default_values[key]:.2f}"))
        for key in ['pm10', 'pm2.5', 'co', 'no', 'no2', 'so2', 'o3']
    }






tabs = st.tabs(["📈 Forecast & Custom Estimation", "Correlation Graphs"])

with tabs[0]:
    df = load_aqi_data()
    df_aqi = df[['ds', 'aqi']].dropna()

    st.subheader("AQI Data")
    st.write(df_aqi.tail())

    st.subheader("📉 AQI Time Series Overview")
    st.line_chart(df_aqi.set_index('ds')['aqi'])

    st.subheader(f"Forecasting AQI for the Next {forecast_days} Days")
    prophet_df = df_aqi.rename(columns={'aqi': 'y'})
    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)

    st.subheader("📈 AQI Forecast")
    fig1 = plot_plotly(model, forecast)
    st.plotly_chart(fig1)

    if enable_manual:
        st.subheader("Custom AQI Forecast from 12:00 to Midnight")
        noon_time = datetime.datetime.combine(selected_date, datetime.time(hour=12))

        manual_row = {'ds': noon_time, **manual_input}
        manual_df = pd.DataFrame([manual_row])

        df_multi = df[['ds', 'aqi', 'pm10', 'pm2.5', 'co', 'no', 'no2', 'so2', 'o3']].dropna().rename(columns={'aqi': 'y'})
        model_manual = Prophet()
        for col in ['pm10', 'pm2.5', 'co', 'no', 'no2', 'so2', 'o3']:
            model_manual.add_regressor(col)
        model_manual.fit(df_multi)

        future_times = pd.date_range(start=noon_time, end=noon_time.replace(hour=23), freq='H')
        future_manual = pd.DataFrame({'ds': future_times})
        for col in ['pm10', 'pm2.5', 'co', 'no', 'no2', 'so2', 'o3']:
            future_manual[col] = manual_input[col]

        forecast_manual = model_manual.predict(future_manual)

        st.subheader("Hourly AQI Forecast")
        fig3 = plot_plotly(model_manual, forecast_manual)
        st.plotly_chart(fig3)

        st.subheader("Forecast Table")
        st.dataframe(forecast_manual[['ds', 'yhat']].rename(columns={'ds': 'Time', 'yhat': 'Predicted AQI'}))

        end_of_day_aqi = forecast_manual['yhat'].iloc[-1]
        st.success(f"Estimated AQI at Midnight: **{end_of_day_aqi:.2f}**")




with tabs[1]:
    st.header("🌤️ AQI vs Weather")

    df_weather = load_weather_data()
    df_all = pd.merge(load_aqi_data(), df_weather, on="ds", how="inner")

    st.subheader("🌡️ AQI vs Temperature")
    fig_temp = px.line(df_all, x='ds', y=['aqi', 'temperature_2m'], title="AQI and Temperature Over Time", labels={'ds': 'Time', 'value': 'Value'})
    st.plotly_chart(fig_temp)

    st.subheader("💧 AQI vs Relative Humidity")
    fig_rh = px.line(df_all, x='ds', y=['aqi', 'relative_humidity_2m'], title="AQI and Relative Humidity Over Time", labels={'ds': 'Time', 'value': 'Value'})
    st.plotly_chart(fig_rh)

    st.subheader("🌧️ AQI vs Precipitation")
    fig_prec = px.line(df_all, x='ds', y=['aqi', 'precipitation'], title="AQI and Precipitation Over Time", labels={'ds': 'Time', 'value': 'Value'})
    st.plotly_chart(fig_prec)

    st.subheader("Trends")
    fig2 = plot_components_plotly(model, forecast)
    st.plotly_chart(fig2)

    




    st.subheader("Time-Lagged Correlation with AQI")
    lags = list(range(0, 25))

    for col in ['temperature_2m', 'relative_humidity_2m', 'weather_code', 'precipitation']:
        corr_values = compute_time_lagged_correlation(df_all, col, lags)
        fig_corr = go.Figure()
        fig_corr.add_trace(go.Scatter(x=lags, y=corr_values, mode='lines+markers'))
        fig_corr.update_layout(title=f"AQI vs {col} - Time-Lagged Correlation",
                               xaxis_title="Lag (hours)", yaxis_title="Correlation Coefficient")
        st.plotly_chart(fig_corr)

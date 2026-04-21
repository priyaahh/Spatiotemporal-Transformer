import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import plotly.graph_objects as go
from datetime import timedelta

import config
from preprocessing import load_and_clean_data, autoregressive_forecast
from risk_analysis import calculate_heat_risk

# --- Configuration ---
st.set_page_config(page_title="Weather Forecast & Heat Risk", layout="wide")
st.title("🌦️ Weather Forecast and Heat Risk Prediction Tool")
st.markdown("Upload your weather dataset to forecast temperatures into the future and evaluate associated heat risks.")

# --- Load Models & Scaler ---
@st.cache_resource
def load_models_and_scaler():
    try:
        lstm = tf.keras.models.load_model(os.path.join(config.MODEL_DIR, "lstm_model.keras"))
        transformer = tf.keras.models.load_model(os.path.join(config.MODEL_DIR, "transformer_model.keras"))
        scaler = joblib.load(os.path.join(config.MODEL_DIR, "scaler.pkl"))
        return lstm, transformer, scaler
    except Exception as e:
        st.error("Models or Scaler not found. Please run `python train.py` to train the models first.")
        return None, None, None

lstm_model, transformer_model, scaler = load_models_and_scaler()

# --- Sidebar UI ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload Weather CSV", type=['csv'])

st.sidebar.header("2. Prediction Settings")
model_choice = st.sidebar.radio("Select Model:", ("LSTM", "Transformer"))
horizon = st.sidebar.selectbox("Forecasting Horizon (Days):", config.SUPPORTED_HORIZONS, index=0)

run_button = st.sidebar.button("Run Prediction", type="primary")

st.sidebar.markdown("---")
st.sidebar.markdown("**Note on Architecture:**")
st.sidebar.markdown("Models are designed modularly. If spatio-temporal imagery (e.g. satellite data) becomes available, models can be extended without altering the core pipeline.")

# --- Process Data & Preview ---
try:
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.info("Using uploaded dataset.")
    else:
        # Fall back to default
        if os.path.exists(config.DATA_FILE):
            raw_df = pd.read_csv(config.DATA_FILE)
            st.info("No file uploaded. Using default `weatherHistory.csv` dataset.")
        else:
            st.error(f"Default dataset `{config.DATA_FILE}` not found. Please upload a file.")
            st.stop()
except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()

# Validate columns
required_cols = ['Formatted Date', 'Temperature (C)', 'Humidity', 'Wind Speed (km/h)']
missing_cols = [col for col in required_cols if col not in raw_df.columns]

if missing_cols:
    st.error(f"Upload failed. The CSV is missing required columns: {missing_cols}")
    st.stop()

# Short preview of raw data
st.subheader("Dataset Preview")
col1, col2, col3 = st.columns(3)
col1.metric("Rows", raw_df.shape[0])
col2.metric("Columns", raw_df.shape[1])
st.dataframe(raw_df.head(), use_container_width=True)

# Clean dataset
try:
    df = load_and_clean_data(raw_df)
except Exception as e:
    st.error(f"Error cleaning dataset: {e}")
    st.stop()

# Validate dataset size
if len(df) < config.SEQ_LENGTH:
    st.warning(f"Dataset is too small to perform predictions. We need at least {config.SEQ_LENGTH} rows of historical data.")
    st.stop()

# --- Run Prediction Flow ---
if run_button:
    st.markdown("---")
    if not lstm_model or not transformer_model or not scaler:
        st.error("Cannot predict without trained models.")
    else:
        with st.spinner(f"Running {model_choice} prediction for the next {horizon} days..."):
            
            # 1. Prepare initial sequence
            recent_data = df[config.FEATURES].tail(config.SEQ_LENGTH)
            scaled_recent = scaler.transform(recent_data)
            initial_sequence = np.expand_dims(scaled_recent, axis=0)
            
            active_model = lstm_model if model_choice == "LSTM" else transformer_model
            
            # 2. Forecast
            forecasted_temps = autoregressive_forecast(active_model, initial_sequence, horizon, scaler)
            
            last_date = df.index[-1]
            future_dates = [last_date + timedelta(days=i) for i in range(1, horizon + 1)]
            
            # 3. Risk Analysis
            historical_temps = df.tail(3)[config.TARGET_COL].values
            risk_report = calculate_heat_risk(forecasted_temps, historical_temps)
            
            # --- Display Results ---
            st.header("Prediction Results")
            
            # Top-level Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Selected Model", model_choice)
            m2.metric("Horizon", f"{horizon} Days")
            m3.metric("Avg Predicted Temp", f"{np.mean(forecasted_temps):.2f} °C")
            
            st.markdown("---")
            
            # Risk Display
            risk_color = "#28a745" # green
            if risk_report["Risk Level"] == "High":
                risk_color = "#dc3545" # red
            elif risk_report["Risk Level"] == "Medium":
                risk_color = "#ffc107" # warning/orange
                
            st.markdown(f"### Heat Risk Level: <span style='color:{risk_color};'>{risk_report['Risk Level']}</span>", unsafe_allow_html=True)
            st.markdown(f"> **Rationale:** {risk_report['Rationale']}")
            st.markdown(f"> **Trend:** {risk_report['Trend']}")
            
            st.markdown("---")
            st.subheader("Temperature Forecast")
            
            # Prepare Results DataFrame
            results_df = pd.DataFrame({
                'Date': [d.strftime('%Y-%m-%d') for d in future_dates],
                'Predicted Temperature (C)': [round(float(t), 2) for t in forecasted_temps],
                'Model': [model_choice] * len(future_dates),
                'Horizon': [horizon] * len(future_dates)
            })
            
            col_data, col_chart = st.columns([1, 2])
            
            with col_data:
                st.dataframe(
                    results_df[['Date', 'Predicted Temperature (C)']], 
                    use_container_width=True,
                    hide_index=True
                )
            
            with col_chart:
                # Create Plotly Graph
                fig = go.Figure()
                
                history_plot_len = min(14, len(df))
                fig.add_trace(go.Scatter(
                    x=df.index[-history_plot_len:], 
                    y=df[config.TARGET_COL].tail(history_plot_len),
                    mode='lines+markers',
                    name='Recent History',
                    line=dict(color='#1f77b4') # blue
                ))
                
                fig.add_trace(go.Scatter(
                    x=future_dates, 
                    y=forecasted_temps,
                    mode='lines+markers',
                    name=f'Forecast ({model_choice})',
                    line=dict(color='#d62728', dash='dash') # red
                ))
                
                fig.update_layout(
                    title=f"Forecasted Temperatures (Next {horizon} Days)",
                    xaxis_title="Date",
                    yaxis_title="Temperature (°C)",
                    template="plotly_white",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # --- Download Results ---
            st.subheader("Export Center")
            st.markdown("Download the complete forecast dataset, including model details and parameters.")
            
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Forecast CSV",
                data=csv,
                file_name=f"forecast_{model_choice.lower()}_{horizon}days.csv",
                mime="text/csv",
                type="primary"
            )

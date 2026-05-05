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


st.set_page_config(
    page_title="Weather Forecast & Heat Risk",
    page_icon="🌡️",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-image: url('https://images.unsplash.com/photo-1499346030926-9a72daac6c63?auto=format&fit=crop&w=1920&q=80');
    background-size: cover;
    background-attachment: fixed;
    background-position: center;
    color: #1f2937;
}

[data-testid="stHeader"] {
    background-color: rgba(255, 255, 255, 0.2) !important;
    backdrop-filter: blur(5px) !important;
}

[data-testid="stSidebar"] {
    background-color: rgba(255, 255, 255, 0.82) !important;
    backdrop-filter: blur(12px);
}

h1 {
    color: #ffffff !important;
    text-shadow: 2px 2px 5px rgba(0,0,0,0.65);
}

h2, h3 {
    color: #1e3a8a !important;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background-color: rgba(255, 255, 255, 0.88);
    backdrop-filter: blur(12px);
    border-radius: 14px;
    padding: 16px;
    box-shadow: 0 8px 18px rgba(0,0,0,0.13);
    border: 1px solid rgba(255,255,255,0.7);
}

div[data-testid="stMetricValue"] {
    color: #db2777;
    font-weight: 800;
}

/* Button */
.stButton>button {
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 22px;
    font-weight: 700;
    box-shadow: 0 5px 14px rgba(37, 99, 235, 0.35);
}

.stButton>button:hover {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    color: white;
    transform: translateY(-1px);
}

/* Better Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 18px;
    background-color: rgba(255, 255, 255, 0.28);
    padding: 8px 12px;
    border-radius: 14px;
}

.stTabs [data-baseweb="tab"] {
    height: 38px;
    padding: 6px 16px;
    border-radius: 999px;
    font-size: 14px;
    background-color: rgba(255, 255, 255, 0.72);
    border: 1px solid rgba(255, 255, 255, 0.55);
    color: #374151;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    background-color: rgba(255, 255, 255, 0.95);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #2563eb, #3b82f6);
    color: white !important;
    box-shadow: 0 5px 14px rgba(37, 99, 235, 0.35);
}

/* Info card */
.info-card {
    background-color: rgba(219, 234, 254, 0.82);
    padding: 18px 22px;
    border-radius: 14px;
    border: 1px solid rgba(147, 197, 253, 0.7);
    margin-bottom: 18px;
    font-size: 16px;
    color: #075985;
}
</style>
""", unsafe_allow_html=True)


st.title("🌅 Advanced Weather Forecast & Heat Risk Prediction")
st.markdown(
    "<p style='font-size:18px; color:#ffffff; font-weight:600; text-shadow:1px 1px 3px rgba(0,0,0,0.8);'>"
    "Unlock future weather insights using deep learning models."
    "</p>",
    unsafe_allow_html=True
)


@st.cache_resource
def load_models_and_scaler():
    try:
        lstm = tf.keras.models.load_model(os.path.join(config.MODEL_DIR, "lstm_model.keras"))
        transformer = tf.keras.models.load_model(os.path.join(config.MODEL_DIR, "transformer_model.keras"))
        scaler = joblib.load(os.path.join(config.MODEL_DIR, "scaler.pkl"))
        return lstm, transformer, scaler
    except Exception:
        return None, None, None


lstm_model, transformer_model, scaler = load_models_and_scaler()

if lstm_model is None or transformer_model is None or scaler is None:
    st.error("⚠️ Models or Scaler not found. Please run `python train.py` first.")


st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload Weather CSV", type=["csv"])

st.sidebar.header("2. Prediction Settings")
model_choice = st.sidebar.radio("Select Neural Network:", ("LSTM", "Transformer"))
horizon = st.sidebar.selectbox("Forecasting Horizon (Days):", config.SUPPORTED_HORIZONS, index=0)

run_button = st.sidebar.button("Run Prediction 🚀", type="primary", width="stretch")

with st.sidebar.expander("ℹ️ Architecture Notes"):
    st.markdown("""
    - **LSTM**: Learns sequential temperature patterns.
    - **Transformer**: Uses attention-based sequence learning.
    """)


tab_data, tab_results, tab_visuals = st.tabs([
    "📂 Data Preparation",
    "📊 Numerical Forecast",
    "🌐 Heat Risk & Charts"
])


raw_df = None
data_source = ""

try:
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        data_source = "Uploaded File"
    else:
        if os.path.exists(config.DATA_FILE):
            raw_df = pd.read_csv(config.DATA_FILE)
            data_source = "Default Dataset"
except Exception as e:
    st.error(f"Error reading dataset: {e}")


df = None

if raw_df is not None:
    required_cols = [
        "Formatted Date",
        "Temperature (C)",
        "Humidity",
        "Wind Speed (km/h)"
    ]

    missing_cols = [col for col in required_cols if col not in raw_df.columns]

    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
    else:
        with tab_data:
            st.subheader("Data Overview")

            st.markdown(
                f"""
                <div class="info-card">
                    <b>Current Data Source:</b> {data_source}
                </div>
                """,
                unsafe_allow_html=True
            )

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Records", f"{len(raw_df):,}")
            col2.metric("Features", raw_df.shape[1])

            last_date = pd.to_datetime(
                raw_df["Formatted Date"],
                errors="coerce"
            ).dropna().max()

            if pd.notna(last_date):
                col3.metric("Last Recorded", last_date.strftime("%Y-%m-%d"))
            else:
                col3.metric("Last Recorded", "N/A")

            with st.expander("👀 View Dataset Snippet"):
                st.dataframe(raw_df.head(100), width="stretch")

            try:
                df = load_and_clean_data(raw_df)

                if len(df) < config.SEQ_LENGTH:
                    st.warning("Not enough data to run sequence prediction.")
                    df = None

            except Exception as e:
                st.error(f"Error cleaning dataset: {e}")

else:
    with tab_data:
        st.warning("Please upload a CSV file or check default dataset path.")


if run_button and df is not None and lstm_model is not None and transformer_model is not None:
    with st.spinner(f"Running prediction using {model_choice}..."):

        recent_data = df[config.FEATURES].tail(config.SEQ_LENGTH)
        scaled_recent = scaler.transform(recent_data)
        initial_sequence = np.expand_dims(scaled_recent, axis=0)

        active_model = lstm_model if model_choice == "LSTM" else transformer_model

        forecasted_temps = autoregressive_forecast(
            active_model,
            initial_sequence,
            horizon,
            scaler
        )

        last_date = df.index[-1]

        future_dates = [
            last_date + timedelta(days=i)
            for i in range(1, horizon + 1)
        ]

        historical_temps = df.tail(3)[config.TARGET_COL].values
        risk_report = calculate_heat_risk(forecasted_temps, historical_temps)

        results_df = pd.DataFrame({
            "Date": [d.strftime("%Y-%m-%d") for d in future_dates],
            "Predicted Temperature (C)": [
                round(float(t), 2) for t in forecasted_temps
            ]
        })

        current_temp = float(df.tail(1)[config.TARGET_COL].values[0])
        avg_pred_temp = float(np.mean(forecasted_temps))
        peak_temp = float(np.max(forecasted_temps))

        with tab_results:
            st.subheader(f"⏱️ {horizon}-Day Forecast ({model_choice})")

            m1, m2, m3 = st.columns(3)

            m1.metric("Current Temp", f"{current_temp:.2f} °C")
            m2.metric(
                "Predicted Avg",
                f"{avg_pred_temp:.2f} °C",
                f"{avg_pred_temp - current_temp:.2f} °C vs Current"
            )
            m3.metric("Peak Predicted Temp", f"{peak_temp:.2f} °C")

            st.dataframe(results_df, width="stretch", hide_index=True)

            csv = results_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="📥 Download Forecast Data",
                data=csv,
                file_name=f"forecast_{model_choice.lower()}.csv",
                mime="text/csv"
            )

        with tab_visuals:
            st.subheader("🌐 Heat Risk Analysis")

            risk_color = "#16a34a"

            if risk_report["Risk Level"] == "High":
                risk_color = "#dc2626"
            elif risk_report["Risk Level"] == "Medium":
                risk_color = "#d97706"

            st.markdown(
                f"""
                <div class="info-card">
                    <h3 style="margin-top:0;">🚨 Detected Heat Risk:
                    <span style="color:{risk_color};">{risk_report["Risk Level"]}</span></h3>
                    <p><b>Rationale:</b> {risk_report["Rationale"]}</p>
                    <p><b>Trend:</b> {risk_report["Trend"]}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Avg Forecast Temp", f"{avg_pred_temp:.2f} °C")
            c2.metric("Peak Forecast Temp", f"{peak_temp:.2f} °C")
            c3.metric("Risk Level", risk_report["Risk Level"])

            fig = go.Figure()
            history_plot_len = min(21, len(df))

            fig.add_trace(go.Scatter(
                x=df.index[-history_plot_len:],
                y=df[config.TARGET_COL].tail(history_plot_len),
                mode="lines+markers",
                name="Recent History",
                line=dict(
                    color="#3b82f6",
                    width=3,
                    shape="spline"
                ),
                marker=dict(size=6)
            ))

            fig.add_trace(go.Scatter(
                x=future_dates,
                y=forecasted_temps,
                mode="lines+markers",
                name=f"Forecast ({model_choice})",
                line=dict(
                    color="#ef4444",
                    width=3,
                    dash="dot",
                    shape="spline"
                ),
                marker=dict(size=8, symbol="diamond")
            ))

            fig.update_layout(
                title="Temperature Forecast Trend",
                xaxis_title="Date",
                yaxis_title="Temperature (°C)",
                template="plotly_white",
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=20, r=20, t=50, b=20),
                height=560,
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=7, label="1w", step="day", stepmode="backward"),
                            dict(count=14, label="2w", step="day", stepmode="backward"),
                            dict(step="all")
                        ])
                    ),
                    rangeslider=dict(visible=True),
                    type="date"
                )
            )

            st.plotly_chart(fig, width="stretch")

elif run_button and df is None:
    st.error("Cannot run prediction. Please check your dataset format.")
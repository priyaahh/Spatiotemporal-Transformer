# 🌦️ Weather Forecast and Heat Risk Prediction Tool

A comprehensive, modular Streamlit application for forecasting future temperatures and evaluating associated heat risks. This project implements autoregressive time-series forecasting using Deep Learning models (LSTM and a lightweight Transformer), tailored for CPU efficiency and practical usability.

## ✨ Features
- **Multiple Model Architectures**: Choose between LSTM and CPU-optimized Transformer models for forecasting.
- **Dynamic Forecasting Horizons**: Predict temperature trends for the next 7, 14, or 30 days.
- **Heat Risk Assessment**: Automatically categorizes future heat risk (Low, Medium, High) based on recent historical data and forecasted trends.
- **Custom Data Upload**: Bring your own weather CSV data (requires matching target columns) or use the included default `weatherHistory.csv` dataset.
- **Rich Visualizations**: Interactive charts generated with Plotly to compare historical temperatures against forecasted predictions.
- **Export Capabilities**: Download the prediction results as a CSV file for further analysis.

## 🛠️ Project Structure
```
MP/
├── app.py                # Main Streamlit web application
├── train.py              # Script to train and save the deep learning models
├── models.py             # Defines the architecture for LSTM and Transformer models
├── preprocessing.py      # Data loading, cleaning, and sequence generation
├── risk_analysis.py      # Logic for evaluating heat risk severity and rationale
├── config.py             # Global configurations, file paths, and hyperparameters
└── weatherHistory.csv    # Default historical weather dataset
```

## 🚀 Getting Started

### 1. Install Dependencies
Ensure you have Python 3.8+ installed. You will need the following key libraries:
```bash
pip install streamlit pandas numpy tensorflow joblib plotly scikit-learn
```

### 2. Train the Models
Before running the Streamlit app, you must train the models to generate the required `.keras` model files and the data scaler. Run the training script:
```bash
python train.py
```
This script will read `weatherHistory.csv`, train the LSTM and Transformer models, and save them into the `saved_models/` directory.

### 3. Run the App
Launch the Streamlit interface:
```bash
python -m streamlit run app.py
```

## 📊 Using the Tool
1. **Upload or Default**: The app will automatically fall back to `weatherHistory.csv` if no file is uploaded. To use a custom dataset, ensure it contains the columns (`Formatted Date`, `Temperature (C)`, `Humidity`, and `Wind Speed (km/h)`).
2. **Select Parameters**: Use the sidebar to toggle between the **LSTM** and **Transformer** models, and select your forecasting horizon.
3. **Run Prediction**: Click "Run Prediction" to execute the autoregressive pipeline. 
4. **Analyze Results**: View the risk rationale, interact with the Plotly chart, and download the full forecast trace using the Export Center.

## 🔗 Repository
[Heat-Risk-Prediction-System](https://github.com/priyaahh/Spatiotemporal-Transformer)

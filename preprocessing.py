import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import config

def load_and_clean_data(file_or_df=config.DATA_FILE):
    """Loads and preprocesses the dataset from a filepath, file-like object, or DataFrame."""
    if isinstance(file_or_df, pd.DataFrame):
        df = file_or_df.copy()
    else:
        df = pd.read_csv(file_or_df)
        
    df = df[['Formatted Date', 'Temperature (C)', 'Humidity', 'Wind Speed (km/h)']]
    df.columns = ['Date', 'Temp', 'Humidity', 'WindSpeed']
    
    # Parse dates and handle timezone issues
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)
    
    return df

def get_scaler():
    """Returns an instance of the scaler."""
    return MinMaxScaler()

def create_sequences(data, seq_length=config.SEQ_LENGTH):
    """
    Creates sequences of length `seq_length` and target y (predicting next step).
    data: scaled numpy array of shape (N, features)
    returns X (N-seq_length, seq_length, features) and y (N-seq_length, target_feature_idx)
    """
    X, y = [], []
    # Assuming target feature 'Temp' is at index 0 from config.FEATURES
    target_idx = 0 
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length][target_idx])
    return np.array(X), np.array(y)

def autoregressive_forecast(model, initial_sequence, steps, scaler):
    """
    Given a trained model and an initial sequence, forecasts `steps` into the future.
    initial_sequence: shape (1, seq_length, features)
    Returns: list of forecasted temperatures (unscaled)
    """
    current_seq = np.copy(initial_sequence)
    forecasts = []
    
    for _ in range(steps):
        # Predict the next temperature
        next_temp_scaled = model.predict(current_seq, verbose=0)[0, 0]
        
        # Carry forward the last row's other features and update the temp
        last_features = current_seq[0, -1, :].copy()
        last_features[0] = next_temp_scaled  # update temperature
        
        # Append the new prediction, unscaled
        dummy = np.zeros((1, len(config.FEATURES)))
        dummy[0, 0] = next_temp_scaled
        unscaled_temp = scaler.inverse_transform(dummy)[0, 0]
        forecasts.append(unscaled_temp)
        
        # Shift sequence forward
        current_seq = np.roll(current_seq, -1, axis=1)
        current_seq[0, -1, :] = last_features
        
    return forecasts

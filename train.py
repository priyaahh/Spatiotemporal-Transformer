import os
import joblib
import config
from preprocessing import load_and_clean_data, get_scaler, create_sequences
from models import build_lstm_model, build_transformer_model

def train_and_save_models():
    print("Loading data...")
    df = load_and_clean_data()
    
    print("Scaling data...")
    scaler = get_scaler()
    data_scaled = scaler.fit_transform(df[config.FEATURES])
    
    # Save the scaler so the app can use it
    scaler_path = os.path.join(config.MODEL_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    print("Creating sequences...")
    X, y = create_sequences(data_scaled)
    
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # Train LSTM
    print("Training LSTM Model...")
    lstm_model = build_lstm_model(input_shape)
    lstm_model.fit(X_train, y_train, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, validation_data=(X_test, y_test))
    lstm_path = os.path.join(config.MODEL_DIR, "lstm_model.keras")
    lstm_model.save(lstm_path)
    print(f"LSTM model saved to {lstm_path}")
    
    # Train Transformer
    print("Training Transformer Model...")
    transformer_model = build_transformer_model(input_shape)
    transformer_model.fit(X_train, y_train, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, validation_data=(X_test, y_test))
    transformer_path = os.path.join(config.MODEL_DIR, "transformer_model.keras")
    transformer_model.save(transformer_path)
    print(f"Transformer model saved to {transformer_path}")
    
    print("Training Complete!")

if __name__ == "__main__":
    train_and_save_models()

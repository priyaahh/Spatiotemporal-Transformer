import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, MultiHeadAttention, LayerNormalization, Dropout, GlobalAveragePooling1D
import config

def build_lstm_model(input_shape):
    """
    Builds the LSTM model.
    """
    model = Sequential([
        Input(shape=input_shape),
        LSTM(config.LSTM_UNITS),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_transformer_model(input_shape):
    """
    Builds a lightweight Transformer model optimized for CPU.
    """
    inputs = Input(shape=input_shape)
    x = inputs
    
    # Layer 1
    for _ in range(config.TRANSFORMER_LAYERS):
        attn_output = MultiHeadAttention(
            num_heads=config.TRANSFORMER_HEADS, 
            key_dim=config.TRANSFORMER_KEY_DIM
        )(x, x)
        x = LayerNormalization()(x + attn_output)
        
        ffn_output = Dense(config.TRANSFORMER_FF_DIM, activation='relu')(x)
        ffn_output = Dense(input_shape[-1])(ffn_output)
        x = LayerNormalization()(x + ffn_output)
        x = Dropout(config.TRANSFORMER_DROPOUT)(x)
        
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs, outputs)
    
    # Using a slightly higher learning rate or AdamW can sometimes help transformers,
    # but standard Adam with MSE is fine for this regression task.
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

def build_spatio_temporal_placeholder(input_shape_time, input_shape_spatial):
    """
    Future Architecture Hook:
    Combines tabular time-series data with spatial data (e.g., images).
    Currently a placeholder.
    """
    input_time = Input(shape=input_shape_time, name="tabular_input")
    input_spatial = Input(shape=input_shape_spatial, name="spatial_input")
    
    # Time-series branch
    x_time = LSTM(32)(input_time)
    
    # Spatial branch (e.g., CNN)
    x_spatial = tf.keras.layers.Flatten()(input_spatial)
    x_spatial = Dense(32, activation='relu')(x_spatial)
    
    # Combine
    combined = tf.keras.layers.Concatenate()([x_time, x_spatial])
    z = Dense(16, activation='relu')(combined)
    outputs = Dense(1, name="output")(z)
    
    model = Model(inputs=[input_time, input_spatial], outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

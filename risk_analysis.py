import numpy as np

def calculate_heat_risk(predicted_temps, historical_temps):
    """
    Enhances heat risk logic by using realistic temperature thresholds
    and evaluating the temperature trend.
    
    predicted_temps: list or array of forecasted temperatures for the horizon
    historical_temps: the recent historical temperatures (e.g., last 3 days)
    
    Returns: a dict containing risk level, rationale, and trend
    """
    avg_pred = np.mean(predicted_temps)
    max_pred = np.max(predicted_temps)
    
    # Calculate trend
    if len(historical_temps) > 0 and len(predicted_temps) > 0:
        hist_avg = np.mean(historical_temps[-3:]) # Last 3 days
        trend_diff = avg_pred - hist_avg
    else:
        trend_diff = 0
        
    if trend_diff > 1.5:
        trend_str = "Sharply Increasing"
    elif trend_diff > 0.5:
        trend_str = "Increasing"
    elif trend_diff < -0.5:
        trend_str = "Decreasing"
    else:
        trend_str = "Stable"
        
    # Realistic Thresholds in Celsius
    # High risk if max temp exceeds 35C OR average is high and trend is increasing
    if max_pred >= 35 or (avg_pred >= 30 and trend_diff > 1.0):
        risk_level = "High"
        rationale = f"Maximum forecasted temperature reaches {max_pred:.1f}°C. Extreme heat precautions advised."
    elif 25 <= max_pred < 35:
        risk_level = "Medium"
        rationale = f"Forecasts indicate warm conditions (Max: {max_pred:.1f}°C, Trend: {trend_str})."
    else:
        risk_level = "Low"
        rationale = f"Temperatures remain comfortable (Average: {avg_pred:.1f}°C)."
        
    return {
        "Risk Level": risk_level,
        "Trend": trend_str,
        "Max Forecased Temp (°C)": round(max_pred, 1),
        "Average Forecasted Temp (°C)": round(avg_pred, 1),
        "Rationale": rationale
    }

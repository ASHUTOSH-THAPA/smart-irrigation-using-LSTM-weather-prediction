import numpy as np
import pandas as pd 
from tensorflow.keras.models import load_model
from datetime import datetime
import joblib

model=load_model('/Users/ashutoshthapa/Documents/irrigation system/weather_prediction_lstm_model.keras')
temp_scaler = joblib.load('/Users/ashutoshthapa/Documents/irrigation system/temp_scaler.pkl')
humi_scaler = joblib.load('/Users/ashutoshthapa/Documents/irrigation system/humidity_scaler.pkl')
pres_scaler = joblib.load('/Users/ashutoshthapa/Documents/irrigation system/pressure_scaler.pkl')
wind_scaler = joblib.load('/Users/ashutoshthapa/Documents/irrigation system/wind_scaler.pkl')
wind_dir=joblib.load('/Users/ashutoshthapa/Documents/irrigation system/wind_dir.pkl')


def predict_weather(current_weather,previous_bservation=[]):
    sequence=previous_bservation+[current_weather]
    if len(sequence)<7:
        return {"error":f"need {7-len(sequence)} more observation"}
    features=[]    
    for obs in sequence:
        scaled=[
            temp_scaler.transform([[obs['temperature']]])[0][0],
            temp_scaler.transform([[obs['temp_min']]])[0][0],
            temp_scaler.transform([[obs['temp_max']]])[0][0],
            humi_scaler.transform([[obs['humidity']]])[0][0],
            pres_scaler.transform([[obs['pressure']]])[0][0],
            wind_scaler.transform([[obs['wind_gust_speed']]])[0][0],
            wind_dir.transform([[obs['wind_gust_dir']]])[0][0]

        ]
        features.append(scaled)
    input_array = np.array(features).reshape(1, 7, 7)  # (1, timesteps, features)
    prediction = model.predict(input_array)
    pred_temp = temp_scaler.inverse_transform([[prediction[0][0]]])[0][0]
    pred_min_temp = temp_scaler.inverse_transform([[prediction[0][1]]])[0][0]
    pred_max_temp = temp_scaler.inverse_transform([[prediction[0][2]]])[0][0]
    pred_humi = humi_scaler.inverse_transform([[prediction[0][3]]])[0][0]
    pred_pres = pres_scaler.inverse_transform([[prediction[0][4]]])[0][0]
    pred_wind_speed = wind_scaler.inverse_transform([[prediction[0][5]]])[0][0]
    
    will_rain=pred_humi>80 and pred_temp<30
    return {
        'predicted_temperature': round(pred_temp, 1),
        'predicted_humidity': round(pred_humi, 1),
        'will_rain': will_rain,
        'rain_probability': min(100, max(0, int((pred_humi - 70) * 3))) if will_rain else 0,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'max_temperature': round(pred_max_temp,1),
        'min_temperature':round(pred_min_temp,1),
        'wind_speed':round(pred_wind_speed,1),
        'pressure':round(pred_pres,1)
    }


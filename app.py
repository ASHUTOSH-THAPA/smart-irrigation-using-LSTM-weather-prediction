"""flask - handles http request/response and routes
render_templates - rendering html templates(your frontend)
request - to access incoming request data(form submissions,json payloads)
jsonify - converts python dictionaries to proper json responses for api endpoints
sqlite3 - used to store weather predictions, save irrigation decisions and retrieve historical data
datetime - oraganize istorical data by date , used to show  current time  
numpy - handles arrays of weather data and reshape data for model input/output
load_model - specifically loads the saved keras model(weather_prediction_lstm_model.keras)
json - for working json data, parse incoming json request """

from flask import Flask ,render_template,request,jsonify 
import sqlite3
from datetime import datetime 
import numpy as np 
from tensorflow.keras.models import load_model
import json 
import requests
from lstm_prediction_model import predict_weather
from flask import session
import os 
import  secrets

app=Flask(__name__)
app.secret_key=os.environ.get('FLASK_SECRET_KEY') or secrets.token_hex(16) # generate a random secret key for session management
api_key='69a27e9a240c69b2da5bd103a7dd47ef'
base_url='https://api.openweathermap.org/data/2.5/'
CROP_WATER_NEEDS ={
    "wheat": {
        "seedling": 0.3,
        "vegetative": 0.7,
        "flowering": 1.15,
        "fruiting": 1.1,
        "mature": 0.25
    },
    "maize": {
        "seedling": 0.4,
        "vegetative": 0.7,
        "flowering": 1.2,
        "fruiting": 1.15,
        "mature": 0.6
    },
    "corn": {
        "seedling": 0.4,
        "vegetative": 0.7,
        "flowering": 1.2,
        "fruiting": 1.15, 
        "mature": 0.6
    },
    "tomato": {
        "seedling": 0.6,
        "vegetative": 0.9,
        "flowering": 1.05,
        "fruiting": 1.15,
        "mature": 0.8
    },
    "potato": {
        "seedling": 0.5,
        "vegetative": 0.8,
        "flowering": 1.1,
        "fruiting": 1.05,
        "mature": 0.85
    },
    "rice": {
        "seedling": 1.1,
        "vegetative": 1.15,
        "flowering": 1.2,
        "fruiting": 1.15,
        "mature": 0.9
    },
    "cotton": {
        "seedling": 0.4,
        "vegetative": 0.75,
        "flowering": 1.15,
        "fruiting": 1.2,
        "mature": 0.5
    }
}

SOIL_ADJUSTMENT = {
    'Clay': 0.9,    # Clay retains more water
    'Sandy': 1.2,   # Sandy soil needs more water
    'Loamy': 1.0,   # Loamy is ideal
    'Silty': 1.1    # Silty needs slightly more
}

@app.route('/',methods=['GET','POST'])
def dashboard():
    current_weather=None
    prediction=None
    city=None
    water_need=None
    irrigation_recommendation=None
    irrigation_schedule=None
    if request.method == 'POST':
        city = request.form.get('location',session.get('last_city'))
        if city:
            session['last_city']= city
            current_weather=get_current_weather(city)   
            if current_weather is None:
                flash(" could not fetch weather data for the specified city, error")         
            if current_weather:
                prediction=predict_weather(current_weather)
        crop=request.form.get('crop') # get the crop from the form submission
        growth_stage=request.form.get('growth_stage') # get the growth stage from the form submission
        soil_type=  request.form.get('soil_type')   # get the soil type from the form submission
        if crop and growth_stage and soil_type:
            session['crop']=crop
            session['growth_stage']=growth_stage
            session['soil_type']=soil_type
            water_need=calculate_water_needs(crop,growth_stage,soil_type)
            irrigation_recommendation=generate_irrigation_recommendation(water_need,current_weather if current_weather else None)
            irrigation_schedule=generate_irrigation_schedule(water_need,current_weather if current_weather else None)
    else:
        city=session.get('last_city')
        if city:
            session['last_city']=city
            current_weather=get_current_weather(city)
            if current_weather:
                prediction=predict_weather(current_weather)
        crop=session.get('crop')
        growth_stage=session.get('growth_stage')
        soil_type=session.get('soil_type')
        if crop and growth_stage and soil_type:
          water_need=calculate_water_needs(crop,growth_stage,soil_type)
          irrigation_recommendation=generate_irrigation_recommendation(water_need,current_weather if current_weather else None) 
          irrigation_schedule=generate_irrigation_schedule(water_need,current_weather if current_weather else None)

    return render_template('front.html',
        current=current_weather or {},
        prediction=prediction or {},
        city=city or {},
        crop=session.get('crop'),
        growth_stage=session.get('growth_stage'),
        soil_type=session.get('soil_type'),
        irrigation_recommendation=irrigation_recommendation,
        irrigation_schedule=irrigation_schedule,
        water_need=water_need
    )

def get_current_weather(city):
    weather_data=None
        #fetch weather data
    try:    
        url=f"{base_url}weather?q={city}&appid={api_key}&units=metric"
        response=requests.get(url)#request to a url and return a response
        data=response.json()# the result is not json but is instead the result of taking json as input  
        if data.get("cod")==200: 
            weather_data={
                'temperature':data['main']['temp'],
                'feels_like':data['main']['feels_like'],
                'temp_min':data['main']['temp_min'],
                'temp_max':data['main']['temp_max'],
                'humidity':data['main']['humidity'],
                'description':data['weather'][0]['description'],
                'country':data['sys']['country'],
                'wind_gust_dir':data['wind']['deg'],
                'pressure':data['main']['pressure'],
                'wind_gust_speed':data['wind']['speed']
            }
        else:
            print(f" Error fetching weather data: {data.get('message', 'Unknown error')}")    
    except requests.executions.RequestException as e:
        print(f"error fetchinh weather data: {str(e)}")        
    except Execution as e:
        print(f"error fetching weather data: {str(e)}")
    return weather_data  
def calculate_water_needs(crop,growth_stage,soil_type):
    growth_stage = growth_stage.lower() if growth_stage else ''  # default to mature stage if not provided
    crop = crop.lower() if crop else ''  # default to wheat if not provided
    base_need=CROP_WATER_NEEDS.get(crop,{}).get(growth_stage,0) ## get the base water need for the crop and growth stage
    soil_factor=SOIL_ADJUSTMENT.get(soil_type,1.0) ## get the soil factor for the soil type
    adjusted_need=base_need*soil_factor # adjust the water need based on soil type
    return round(adjusted_need,2)
def generate_irrigation_recommendation(water_need,current_weather):
    if not current_weather:
        return f" Apply {water_need} mm of water per week based on crop recommendation."
    temp=current_weather.get('temperature',20)
    humidity=current_weather.get('humidity',50)
    if temp > 30 and humidity <40:
        adjustment=1.2 # increase water need by 20% in hot and dry conditions
    elif temp <15 and humidity >80:
        adjustment=0.8 #decrease water need by 20% in cold and humid conditions
    else:
        adjustment=1.0 # normal conditions
    adjusted_water=water_need * adjustment # adjusted water need based on current weather
    if 'rain' in current_weather['description'].lower():
        return f"Reduce irrigation by {round(adjusted_water*0.7,1)} mm this week due to rain."
    else:
        return f"apply {round(adjusted_water,1)} mm of water this week based on current weather conditions."       
def generate_irrigation_schedule(water_need, current_weather):
    if not current_weather:
        return "No weather data available for irrigation schedule."
    
    temp = current_weather.get('temperature', 20)
    humidity = current_weather.get('humidity', 50)
    
    if temp > 30 and humidity < 40:
        frequency = "twice a week"
    elif temp < 15 and humidity > 80:
        frequency = "once a week"
    else:
        frequency = "every three days"
    
    return f"Water {water_need} mm of water {frequency} based on current weather conditions."
if __name__=='__main__':
    app.run(debug=True, port=5001)
from flask import Flask, render_template, redirect, url_for, request
import joblib
from sklearn.preprocessing import PowerTransformer
import pandas as pd

model = joblib.load("models/AQI_model.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/map')
def map():
    return render_template('static/map.html')

@app.route('/dashboard')
def dashboard():
    return render_template('static/dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    pm25 = float(request.form['pm25'])
    pm10 = float(request.form['pm10'])
    o3 = float(request.form['o3'])
    so2 = float(request.form['so2'])
    co = float(request.form['co'])
    wind_speed = float(request.form['wind_speed'])
    temperature = float(request.form['temperature'])

    # Create a feature vector from the input values
    features = [[pm25, pm10, o3, so2, co, wind_speed, temperature]]

    scaler = PowerTransformer(method='yeo-johnson')

    X_train = pd.read_csv('src/cleaned_data.csv')
    # Assuming X_train is the original training data used for scaling
    scaler.fit(X_train)

    # Apply the fitted scaler to transform the random features
    X_scaled = scaler.transform(features)

    # Make predictions using the loaded model
    predictions = model.predict(X_scaled)
    prediction = predictions[0]
    print("Predictions:", prediction)

    # Make predictions using the loaded model
    predictions = model.predict(X_scaled)

    # Define the AQI range, color code, and possible health impact
    aqi_range = ""
    color_code = ""
    health_impact = ""
    if 0 <= prediction <= 50:
        text = "Good"
        health_impact = "Minimal impact"
    elif 51 <= prediction <= 100:
        text = "Satisfactory"
        health_impact = "Minor breathing discomfort to sensitive people"
    elif 101 <= prediction <= 200:
        text = "Moderate"
        health_impact = "Breathing discomfort to people with lungs, asthma, and heart diseases"
    elif 201 <= prediction <= 300:
        text = "Poor"
        health_impact = "Breathing discomfort to most people on prolonged exposure"
    elif 301 <= prediction <= 400:
        text = "Very Poor"
        health_impact = "Respiratory illness on prolonged exposure"
    elif 401 <= prediction <= 500:
        text = "Severe"
        health_impact = "Affects healthy people and seriously impacts those with existing diseases"
    
    # Return the predicted AQI, color code, and health impact as a response
    return render_template('result.html', aqi=prediction, aqi_range=aqi_range, text=text, health_impact=health_impact)


if __name__ == '__main__':
    app.run(debug=True)

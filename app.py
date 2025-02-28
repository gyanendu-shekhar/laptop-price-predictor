import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)

# Load the pipeline (pipe.pkl) - this includes both the model and any preprocessing steps
model_path = 'pipe.pkl'
with open(model_path, 'rb') as model_file:
    pipeline = pickle.load(model_file)

# Define a function for prediction
def predict_laptop_price(features):
    # Make prediction using the pipeline
    prediction = pipeline.predict([features])
    return prediction[0]

@app.route('/')
def home():
    return render_template('index.html')  # Assuming you have an HTML file for the frontend

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input features from the form
        brand = request.form['brand']
        processor = request.form['processor']
        ram = float(request.form['ram'])
        storage = float(request.form['storage'])
        screen_size = float(request.form['screen_size'])
        weight = float(request.form['weight'])
        
        # Example feature processing (you should update this as per your feature set)
        features = [ram, storage, screen_size, weight]
        
        # Make prediction
        price = predict_laptop_price(features)
        
        return jsonify({'predicted_price': price})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Specify the port (default is 5000, you can change it)
    app.run(debug=True, host='0.0.0.0', port=5000)  # Change the port if needed


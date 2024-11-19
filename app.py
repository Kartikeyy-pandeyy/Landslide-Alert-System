from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from flask_cors import CORS
import os
import boto3
import time
import json
from datetime import datetime

# Enable CORS for all routes
app = Flask(__name__)
CORS(app)

# Load the model (with the new path)
model_path = 'random_forest_model.joblib'
model = joblib.load(model_path)

# AWS S3 Configuration
S3_BUCKET = 'landslide-alert-system'
S3_KEY = 'AKIA6ODUZXRGJLDEPXPL'
S3_SECRET = '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'
S3_REGION = 'us-east-1'

# Initialize S3 client
s3 = boto3.client(
    's3',
    aws_access_key_id=S3_KEY,
    aws_secret_access_key=S3_SECRET,
    region_name=S3_REGION
)

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request
        data = request.json
        print(f"Received data: {data}")  # Log the received data to the console

        # Ensure all required fields are present
        required_fields = [
            'Elevation', 'Slope Gradient', 'Rainfall', 'City',  'Season',
            'Vegetation Cover', 'Soil Moisture', 'Humidity', 'Temperature'
        ]
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400

        # Convert data to DataFrame (make sure columns align with model training)
        df = pd.DataFrame([data])

        # Log data to verify it's correctly formatted
        print(f"Data for prediction: {df}")

        # Make prediction
        prediction = model.predict(df)
        result = "Yes" if prediction[0] == 1 else "No"
        print(f"Prediction: {result}")  # Log the prediction result

        return jsonify({'prediction': result})

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': 'Error making prediction!'}), 500


# Save Emergency data route
@app.route('/save-emergency', methods=['POST'])
def save_emergency():
    try:
        # Get the emergency data from the request
        data = request.json
        print(f"Received emergency data: {data}")

        # Validate required fields
        required_fields = ['name', 'phone', 'location', 'emergencyType', 'details']
        for field in required_fields:
            if field not in data or not data[field].strip():
                return jsonify({'error': f'Missing or empty field: {field}'}), 400

        # Use phone number as filename
        phone_number = data['phone']
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        file_name = f"{phone_number}.json"  # Save file as phone_number_timestamp.json

        # Save data to S3 as a JSON file
        s3.put_object(Bucket=S3_BUCKET, Key=file_name, Body=json.dumps(data))
        print(f"Emergency data saved to {S3_BUCKET}/{file_name}")

        return jsonify({'message': f'Emergency data saved as {file_name}'}), 200

    except Exception as e:
        print(f"Error saving emergency data: {str(e)}")
        return jsonify({'error': 'An error occurred while saving data'}), 500


if __name__ == '__main__':
    app.run(debug=True)

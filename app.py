from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS
import os

# Enable CORS for all routes
app = Flask(__name__)
CORS(app)

# Load the model (with the new path)
model_path = 'random_forest_model.joblib'
model = joblib.load(model_path)

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

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
import logging

# Assuming these are defined in your local modules
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Initialize Flask application
application = Flask(__name__)
CORS(application)  # Allow cross-origin requests
app = application

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        logging.debug("Received POST request for prediction")

        try:
            # Parse JSON data from request
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400

            # Extract gender and category (upper/lower) from the input data
            gender = data.get('gender').lower()
            category = data.get('topBottom').lower()  # Ensure lower case for consistency
            # Validate input data
            if gender not in ['male', 'female']:
                return jsonify({'error': 'Invalid gender provided'}), 400

            if category not in ['top', 'bottom']:
                return jsonify({'error': 'Invalid category provided'}), 400

            # Initialize an empty dictionary to store the attributes
            attributes = {}

            # Determine which attributes to extract based on gender and category
            if category == 'top':
                attributes['sho_gi'] = float(data.get('sho_gi'))
                attributes['che_gi'] = float(data.get('che_gi'))
                attributes['nav_gi'] = float(data.get('nav_gi'))
                attributes['wai_gi'] = float(data.get('wai_gi'))
            elif category == 'bottom':
                attributes['wai_gi'] = float(data.get('wai_gi'))
                attributes['hip_gi'] = float(data.get('hip_gi'))
                attributes['thi_gi'] = float(data.get('thi_gi'))
                attributes['cal_gi'] = float(data.get('cal_gi'))
                attributes['kne_gi'] = float(data.get('kne_gi'))  # Add knee girth
                attributes['ank_gi'] = float(data.get('ank_gi'))  # Add ankle girth

            # Validate attribute values
            print(attributes)
            if not all(attributes.values()):
                return jsonify({'error': 'Some required attributes are missing or invalid.'}), 400

            # Create an instance of CustomData with the extracted data
            custom_data = CustomData(gender=gender, category=category, **attributes)

            # Convert to DataFrame
            pred_df = custom_data.get_data_as_data_frame()
            logging.debug(f"Data received for prediction: \n{pred_df}")

            # Initialize the prediction pipeline
            predict_pipeline = PredictPipeline(gender=gender, category=category)
            logging.debug("Initialized PredictPipeline")

            # Predict
            results = predict_pipeline.predict(pred_df)
            logging.debug(f"Prediction results: {results}")

            # Convert the numerical prediction results to size labels
            if isinstance(results, np.ndarray):
                results = results.tolist()

            size_mapping = {0: "S", 1: "L", 2: "M", 3: "XL"}
            size_label = size_mapping.get(results[0], "Unknown") if results and isinstance(results[0], int) else "No prediction result found."

            # Return prediction result as JSON
            return jsonify({'message': size_label})

        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)

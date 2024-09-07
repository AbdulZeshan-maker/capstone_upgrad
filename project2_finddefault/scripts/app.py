import os
import pickle
from flask import Flask, request, jsonify

# Load the saved model
model_folder_path = "E:\\capstone_upgrad\\project2_finddefault\\models"
model_filename = 'rf_classifier.pkl'  
model_filepath = os.path.join(model_folder_path, model_filename)

with open(model_filepath, 'rb') as file:
    rf_classifier = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the RandomForest Model API!"

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request
    data = request.json
    # Convert data to the format required by the model
    features = [data['feature1'], data['feature2'], ...]  # Add all required features here

    # Make prediction using the loaded model
    prediction = rf_classifier.predict([features])
    
    # Return the prediction result
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)

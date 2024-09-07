from flask import Flask, request, render_template, redirect, url_for
import os
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and music data
model = load_model('mood_classification_model.h5')
music_data = pd.read_csv('data_moods.csv')

# Define mood labels
mood_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Create a folder to store uploaded images
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Route to upload page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        # Save the uploaded image to the uploads folder
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Load and preprocess the image
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (48, 48))
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=-1)
        image = image / 255.0  # Normalize the image

        # Predict the mood
        predicted_class = model.predict(image)
        predicted_mood = np.argmax(predicted_class)
        predicted_mood_label = mood_labels[predicted_mood]

        # Recommend music based on the predicted mood
        recommendations = music_data[music_data['mood'].str.lower() == predicted_mood_label]
        recommended_songs = recommendations[['name', 'artist']].sample(5).to_dict(orient='records')

        return render_template('result.html', mood=predicted_mood_label, songs=recommended_songs)

# Run the app
if __name__ == '__main__':
    # In production, remove debug=True and use a production server like gunicorn
    app.run(host='0.0.0.0', port=5000)

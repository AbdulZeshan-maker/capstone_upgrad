# predict.py
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import pandas as pd

# Load the music data
music_data = pd.read_csv('/mnt/data/data_moods.csv')

def predict_mood_and_recommend_music(image_path):
    # Load and preprocess the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (48, 48))
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    image = image / 255.0  # Normalize the image
    
    # Load the trained model
    model = load_model('mood_classification_model.h5')
    
    # Predict the mood
    predicted_class = model.predict(image)
    predicted_mood = np.argmax(predicted_class)
    
    # Map the predicted mood to the actual labels
    mood_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
    predicted_mood_label = mood_labels[predicted_mood]
    
    # Recommend music based on the predicted mood
    recommendations = music_data[music_data['mood'].str.lower() == predicted_mood_label]
    recommended_songs = recommendations[['name', 'artist']].sample(5)
    
    return predicted_mood_label, recommended_songs

if __name__ == "__main__":
    # Example Usage
    image_path = "path/to/your/image.png"  # Replace with an actual image path
    mood, songs = predict_mood_and_recommend_music(image_path)
    print(f"Predicted Mood: {mood}")
    print("Recommended Songs:")
    print(songs)

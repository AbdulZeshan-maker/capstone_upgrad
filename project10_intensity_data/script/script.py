# Step 1: Import Libraries
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Step 2: Load and Combine the Data
anger_df = pd.read_csv('intensity_data/angriness.csv')
happiness_df = pd.read_csv('intensity_data/happiness.csv')
sadness_df = pd.read_csv('intensity_data/sadness.csv')

# Label the data according to emotion type
anger_df['label'] = 'angriness'
happiness_df['label'] = 'happiness'
sadness_df['label'] = 'sadness'

# Combine the data
df = pd.concat([anger_df, happiness_df, sadness_df], ignore_index=True)

# Display sample data
df.head()

# Step 3: Data Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to clean text
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)  # Remove non-alphabetical characters
    text = text.lower()  # Lowercase the text
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

# Apply the preprocessing function
df['clean_content'] = df['content'].apply(preprocess_text)

# Encode labels into numeric values for classification
df['label_num'] = df['label'].map({'angriness': 0, 'happiness': 1, 'sadness': 2})

# Step 4: Feature Engineering
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['clean_content']).toarray()
y = df['label_num']

# Step 5: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Model Selection and Training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print(classification_report(y_test, y_pred, target_names=['angriness', 'happiness', 'sadness']))

# Step 7: Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best parameters and evaluate the tuned model
best_model = grid_search.best_estimator_
y_pred_tuned = best_model.predict(X_test)

# Evaluate the tuned model
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
print(f'Tuned Model Accuracy: {accuracy_tuned * 100:.2f}%')
print(classification_report(y_test, y_pred_tuned, target_names=['angriness', 'happiness', 'sadness']))

# Step 8: Save the Model for Deployment
import pickle

# Save the model as a pickle file
with open('intensity_model.pkl', 'wb') as model_file:
    pickle.dump(best_model, model_file)

# Save the vectorizer
with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

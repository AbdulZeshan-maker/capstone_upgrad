# Step 1: Importing Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import pickle

# Step 2: Loading Data
train_data = pd.read_excel('/mnt/data/train.xlsx')
test_data = pd.read_excel('/mnt/data/test.xlsx')
print("Train Data Overview:")
print(train_data.head())
print("\nTest Data Overview:")
print(test_data.head())

# Step 3: Exploratory Data Analysis (EDA)
print("Missing values in train dataset:")
print(train_data.isnull().sum())
print("\nSummary statistics of train dataset:")
print(train_data.describe())
plt.figure(figsize=(6,4))
sns.countplot(x='target_column', data=train_data)  # Replace 'target_column' with actual target column
plt.title('Target Class Distribution')
plt.show()

# Step 4: Data Cleaning
train_data.fillna(method='ffill', inplace=True)
test_data.fillna(method='ffill', inplace=True)
train_data['date_column'] = pd.to_datetime(train_data['date_column'])  # Replace 'date_column' accordingly
columns_to_drop = ['irrelevant_column1', 'irrelevant_column2']  # Replace with actual irrelevant columns
train_data.drop(columns=columns_to_drop, inplace=True)
test_data.drop(columns=columns_to_drop, inplace=True)

# Step 5: Dealing with Imbalanced Data
X = train_data.drop('target_column', axis=1)  # Replace 'target_column' with actual target column
y = train_data['target_column']
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
plt.figure(figsize=(6,4))
sns.countplot(x=y_res)
plt.title('Balanced Target Class Distribution')
plt.show()

# Step 6: Feature Engineering and Selection
scaler = StandardScaler()
X_res_scaled = scaler.fit_transform(X_res)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Step 7: Train-Test Split
X_train, X_valid, y_train, y_valid = train_test_split(X_res_scaled, y_res, test_size=0.2, random_state=42)

# Step 8: Model Training
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}
for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    print(f"\n{model_name} Evaluation:")
    print(f"Accuracy: {accuracy_score(y_valid, y_pred):.2f}")
    print(classification_report(y_valid, y_pred))
    print(confusion_matrix(y_valid, y_pred))

# Step 9: Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")

# Step 10: Testing on Unseen Data
test_data_scaled = scaler.transform(test_data.drop('target_column', axis=1))  # Drop same irrelevant columns
best_model = grid_search.best_estimator_
test_predictions = best_model.predict(test_data_scaled)
test_data['Prediction'] = test_predictions
test_data.to_csv('test_with_predictions.csv', index=False)

# Step 11: Model Deployment Plan
with open('propensity_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("Model saved for deployment.")

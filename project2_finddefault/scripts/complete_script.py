import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE

# Load the dataset
df = pd.read_csv('creditcard.csv')

# Display the first few rows
df.head()

# Exploratory Data Analysis (EDA)
# Data Quality Check and Handling Missing Values
print(df.isnull().sum())
print(df.describe())
print(df.dtypes)

# Visualization of Fraudulent and Non-Fraudulent Transactions
plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df)
plt.title('Distribution of Fraudulent vs Non-Fraudulent Transactions')
plt.show()

# Checking the percentage of fraudulent transactions
fraud_percentage = df['Class'].value_counts(normalize=True) * 100
print(fraud_percentage)

# Outliers visualization for amount
plt.figure(figsize=(10,6))
sns.boxplot(x='Class', y='Amount', data=df)
plt.ylim(0, 500)
plt.title('Boxplot of Transaction Amount by Class')
plt.show()

# Check the datatype for Time column and convert if necessary
df['Time'] = pd.to_datetime(df['Time'], unit='s', origin='unix')
df['Hour'] = df['Time'].dt.hour
df.drop(['Time'], axis=1, inplace=True)

# Separate features and target variable
X = df.drop('Class', axis=1)
y = df['Class']

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Scale numerical columns
scaler = StandardScaler()
numeric_columns = ['Amount', 'Hour']
X_resampled[numeric_columns] = scaler.fit_transform(X_resampled[numeric_columns])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train a RandomForest Classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
roc_auc = roc_auc_score(y_test, rf_classifier.predict_proba(X_test)[:, 1])
print("ROC AUC Score:", roc_auc)

# Plot Feature Importances
feature_importances = rf_classifier.feature_importances_
importances = pd.Series(feature_importances, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
importances.plot(kind='bar')
plt.title('Feature Importances')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.show()

# Save the model
import os
import pickle
model_folder_path = "E:\\capstone_upgrad\\project2_finddefault\\models"
model_filename = 'rf_classifier_model.pkl'
model_filepath = os.path.join(model_folder_path, model_filename)

with open(model_filepath, 'wb') as file:
    pickle.dump(rf_classifier, file)

print(f"Model saved to {model_filepath}")



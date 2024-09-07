import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
# Load the data from the Excel file
file_path = "E:\DATA SCIENCE\capstone\datasets\Copy of AnomaData.xlsx"
data = pd.read_excel(file_path)
# Data quality check
print(data.info())
print(data.describe())

# Visualize missing values
plt.figure(figsize=(10, 6))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

# Handling missing values
data = data.fillna(method='ffill')

# Visualize target variable distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='y', data=data)
plt.title("Distribution of Target Variable 'y'")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(data.corr(), cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap")
plt.show()
# Separate features and target variable
X = data.drop('y', axis=1)
y = data['y']

# Select only numeric features for feature selection
X_numeric = X.select_dtypes(include=[float, int])

# Apply feature selection using ANOVA F-test
selector = SelectKBest(score_func=f_classif, k='all')
X_new = selector.fit_transform(X_numeric, y)

# Get selected feature names
selected_features = X_numeric.columns[selector.get_support(indices=True)]
print(f"Selected Features: {selected_features}")

# Update the dataframe with selected features and target variable
data_selected = data[selected_features]
data_selected['y'] = y
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data_selected.drop('y', axis=1), data_selected['y'], test_size=0.3, random_state=42)

# Verify the split
print(f"Train set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
# Initialize RandomForestClassifier
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time

# Reduced parameter grid for quicker testing
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5]
}

# Start time for debugging
start_time = time.time()

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_train)

# End time for debugging
end_time = time.time()
print(f"GridSearchCV took {end_time - start_time:.2f} seconds.")

# Best parameters and best score
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-validation Score: {grid_search.best_score_}")

# Update the model with the best parameters
best_model = grid_search.best_estimator_

# Evaluate the best model on test data
y_pred_best = best_model.predict(X_test)
print("Best Model Classification Report:")
print(classification_report(y_test, y_pred_best))
print("Best Model Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_best))

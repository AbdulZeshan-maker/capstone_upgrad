# preprocessing.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif

def load_data(file_path):
    """Load the data from the Excel file."""
    data = pd.read_excel(file_path)
    print(data.info())
    print(data.describe())
    return data

def visualize_missing_values(data):
    """Visualize missing values in the dataset."""
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
    plt.title("Missing Values Heatmap")
    plt.show()

def handle_missing_values(data):
    """Handle missing values by forward filling."""
    return data.fillna(method='ffill')

def visualize_target_distribution(data, target_column='y'):
    """Visualize the distribution of the target variable."""
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target_column, data=data)
    plt.title(f"Distribution of Target Variable '{target_column}'")
    plt.show()

def visualize_correlation_heatmap(data):
    """Visualize the correlation heatmap of the dataset."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(data.corr(), cmap='coolwarm', annot=False)
    plt.title("Correlation Heatmap")
    plt.show()

def select_features(X, y):
    """Select features using ANOVA F-test."""
    selector = SelectKBest(score_func=f_classif, k='all')
    X_new = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support(indices=True)]
    print(f"Selected Features: {selected_features}")
    return X_new, selected_features

def split_data(data, target_column='y', test_size=0.3, random_state=42):
    """Split the dataset into training and testing sets."""
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"Train set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test

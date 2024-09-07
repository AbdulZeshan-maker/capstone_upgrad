# preprocessing.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """
    Load dataset from a CSV file.
    """
    df = pd.read_csv(filepath)
    return df

def eda(df):
    """
    Perform exploratory data analysis on the dataset.
    """
    # Data Quality Check and Handling Missing Values
    print("Missing values:\n", df.isnull().sum())
    print("Basic statistics:\n", df.describe())
    print("Data types:\n", df.dtypes)
    
    # Visualization of Fraudulent and Non-Fraudulent Transactions
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Class', data=df)
    plt.title('Distribution of Fraudulent vs Non-Fraudulent Transactions')
    plt.show()
    
    fraud_percentage = df['Class'].value_counts(normalize=True) * 100
    print(f"Fraudulent transactions percentage:\n{fraud_percentage}")
    
    # Outliers visualization for amount
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Class', y='Amount', data=df)
    plt.ylim(0, 500)
    plt.title('Boxplot of Transaction Amount by Class')
    plt.show()

def preprocess_data(df):
    """
    Preprocess the dataset: handle outliers, convert data types, etc.
    """
    # Convert 'Time' column to numeric features (extracting hour)
    df['Time'] = pd.to_datetime(df['Time'], unit='s', origin='unix')
    df['Hour'] = df['Time'].dt.hour
    df.drop(['Time'], axis=1, inplace=True)
    
    # Separate features and target variable
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Apply SMOTE to balance the dataset
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Scale numerical columns in the data
    scaler = StandardScaler()
    numeric_columns = ['Amount', 'Hour']
    X_resampled[numeric_columns] = scaler.fit_transform(X_resampled[numeric_columns])
    
    return X_resampled, y_resampled

def split_data(X, y):
    """
    Split the data into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

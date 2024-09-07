capstone_upgrad folder contains following folders:
1. data
2. models
3. notebook
4. scripts
5. visuals
6. requirements.txt

data contains following folders :
1. raw 
2. processed

scripts contains
1. app.py
2. complete_script.py
3. preprocessing.py

you can pass the follwing instruction to preprocess.py if you want to use it
as function and those instructions are 
# Load the dataset
df = load_data('creditcard.csv')

# Perform EDA
eda(df)

# Preprocess the data
X_resampled, y_resampled = preprocess_data(df)

# Split the data
X_train, X_test, y_train, y_test = split_data(X_resampled, y_resampled)
as we have been guided in the project description:
Data Preprocessing: Include any data preprocessing steps as separate functions or modules in your
project. This ensures reproducibility during deployment.

app.py will help you with the deployment of trained model

model folder contains trained m,odel

In case of any doubt please refer project_2.ipynb file it has complete code rest folder wise
is given for your ease




# Credit Card Fraud Detection

This project is designed to detect fraudulent transactions from a dataset of credit card transactions. It involves data preprocessing, exploratory data analysis (EDA),
feature engineering, and training a machine learning model using a Random Forest classifier.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Results](#results)


## Overview

The aim of this project is to build a machine learning model that can accurately identify fraudulent credit card transactions. The model is trained on a dataset
with imbalanced classes, where the majority of transactions are legitimate, and only a small fraction are fraudulent. The project applies SMOTE for handling class 
imbalance and uses a Random Forest classifier to make predictions.

## Dataset

datset both raw and preprocessed is available in data folder



if you want to use just preprocess data , you can use it as a function, but if you want whole script , then complete script is available
model folder contains pkl trained model for deployment

project structure is given at starting

**Install required packages**:
    Create a virtual environment and install the required packages using the `requirements.txt` file.

1. **Data Preprocessing**: Run the `preprocess.py` script to clean and preprocess the data. This script will generate the preprocessed data file 

    ``` but complete preprocessed data code with data been completely preprocessed is given in notebook folder and in complete script folder

2. **Model Training**: Use the Jupyter notebook in the `notebook/` folder to train the machine learning model. The notebook includes steps for EDA,
 feature engineering, model training, and evaluation.

3. **Model Deployment**: To deploy the model, run the `app.py` instruction
## Model Training

The model training process is documented in the Jupyter notebook located in the `notebook/` folder. This notebook walks through the steps of data exploration,
 preprocessing, feature engineering, model training, and evaluation.

## Results

The Random Forest classifier achieved a high level of accuracy in detecting fraudulent transactions. The evaluation metrics used include:

- **Classification Report**: Precision, Recall, F1-Score for both classes.
- **Confusion Matrix**: Visualization of the true positive, false positive, true negative, and false negative rates.
- **ROC AUC Score**: The area under the receiver operating characteristic curve, indicating the model's ability to distinguish between the classes.

Instructions for Deployment
Install Dependencies: Make sure you have Flask and other necessary libraries installed. You can use the requirements.txt file for this purpose.

Flask
pandas
numpy
scikit-learn
imbalanced-learn
Install the dependencies using:
requirements.txt
Prepare the Model: trained model is saved in the correct path specified in app.py.
The path used in the script is 'E:\\capstone_upgrad\\project2_finddefault\\models\\rf_classifier_model.pkl'. Adjust this path if necessary.

Run the Flask Application: Navigate to the folder containing app.py and run the Flask application with:


python app.py
By default, Flask will start the application on http://127.0.0.1:5000/.

Test the API: You can test the API using tools like curl, Postman, or by writing a simple client script. Here’s an example of how you might test it with curl:


curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"Amount": 123.45, "Hour": 14}'
Replace the JSON data with actual feature values to match your model's expected input.

Deploy to a Cloud Service (Optional): If you want to deploy the Flask application to a cloud service like Heroku or AWS, follow the respective service’s deployment 
instructions. You'll need to create additional files like Procfile for Heroku or Dockerfiles for containerized deployment.

For Heroku:

Create a Procfile with the following content:
makefile

web: python app.py
Install the Heroku CLI and deploy your application:

heroku create
git push heroku master

For Docker:

Create a Dockerfile with the following content:
dockerfile

# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py"]
Build and run the Docker container:

docker build -t credit-card-fraud-detection .
docker run -p 5000:5000 credit-card-fraud-detection
This setup allows you to deploy your model and make predictions through a simple web API.




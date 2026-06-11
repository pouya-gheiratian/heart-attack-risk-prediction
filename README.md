# Heart Attack Risk Prediction

A Machine Learning project that predicts the risk of heart attack using patient medical information and a Decision Tree Classifier.

## Project Overview

This project aims to classify patients into low-risk and high-risk categories for heart attack based on medical and clinical features.

The model is trained using a Decision Tree Classifier and can predict the risk level of a new patient by receiving medical information as input.

## Dataset Description

The dataset contains medical records and health indicators of patients.

### Features

* Age
* Sex
* Chest Pain Type (cp)
* Resting Blood Pressure (trtbps)
* Cholesterol Level (chol)
* Fasting Blood Sugar (fbs)
* Resting ECG Results (restecg)
* Maximum Heart Rate Achieved (thalachh)
* Exercise Induced Angina (exng)
* Old Peak
* Slope (slp)
* Number of Major Vessels (caa)
* Thal (thall)

### Target Variable

* 0 → Low Risk of Heart Attack
* 1 → High Risk of Heart Attack

## Data Processing

* Dataset loading using Pandas
* Feature and target separation
* Train-Test Split (80% Training, 20% Testing)
* Data preparation for classification

## Machine Learning Model

Decision Tree Classifier

Model Parameters:

* Criterion: Entropy
* Max Depth: 5
* Random State: 42

## Model Evaluation

The model performance is evaluated using:

* Accuracy Score
* F1 Score
* Confusion Matrix
* Classification Report

## Prediction System

The application allows users to enter patient information manually and receive:

* Predicted Heart Attack Risk
* Probability of Low Risk
* Probability of High Risk

## Technologies Used

* Python
* Pandas
* Scikit-Learn

## Objective

The objective of this project is to demonstrate the application of machine learning classification techniques in healthcare-related prediction tasks.

## Disclaimer

This project is intended for educational and machine learning purposes only. It should not be used as a substitute for professional medical diagnosis or healthcare advice.

## Author

Mohammad Pouya Gheiratian

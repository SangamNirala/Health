use  streamlit run streamlit_app.py to run the file
# Health
LINK to google colab https://colab.research.google.com/drive/1kAlFJ4vw4_aNoMzL9YrhXekZ2KWljpNC?usp=sharing
![Accuracy](https://github.com/SangamNirala/Health/blob/main/accuracy.jpg)

![Accuracy](https://github.com/SangamNirala/Health/blob/main/prediction%20image.jpg)

# Health Prediction App

## Overview
This is a Streamlit-based web application that predicts an individual's health status based on their lifestyle and health metrics. The app uses machine learning to make predictions based on user input.

## Features
- User-friendly interface for inputting health and lifestyle data
- Real-time health status prediction
- Handles various health metrics including:
  - Physical fitness
  - Diet preference
  - Activity level
  - Sleep hours
  - Mindfulness practice
  - Career
  - Gender
  - Daily steps
  - Daily calories

## How to Run
1. Install required packages:
   ```bash
   pip install streamlit pandas scikit-learn
   ```
2. Run the app:
   ```bash
   streamlit run streamlit_app.py
   ```
3. Open the provided URL in your web browser

## Input Parameters
- **Physical Fitness**: Scale from 1 to 10
- **Diet Preference**: Vegan, Vegetarian, Non-Vegetarian
- **Activity Level**: Sedentary, Lightly Active, Active, Highly Active
- **Sleep Hours**: 0 to 12 hours
- **Mindfulness Practice**: Scale from 1 to 10
- **Career**: Artist, Teacher, Freelancer, Doctor, Business, Nurse, Lawyer, Scientist, Engineer, Manager
- **Gender**: Male, Female
- **Daily Average Steps**: Number of steps
- **Daily Average Calories**: Calorie intake

## Model Details
- Uses Logistic Regression for binary classification
- Trained on health and lifestyle data
- Handles missing values by dropping incomplete records
- Preprocessing includes:
  - One-hot encoding for categorical variables
  - Standard scaling for numerical features

## Prediction Output
The app will display one of two results:
- **Healthy**
- **Not Healthy**

## Requirements
- Python 3.7+
- Streamlit
- Pandas
- Scikit-learn

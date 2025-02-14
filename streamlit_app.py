import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load and preprocess data
@st.cache_data
def load_data():
    data = pd.read_csv('innovize_final_ml.csv')
    # Drop rows with missing values
    data = data.dropna()
    return data

# Train model
@st.cache_data
def train_model(data):
    # Define features and target
    X = data.drop('is_healthy', axis=1)
    y = data['is_healthy']
    
    # Preprocessing
    categorical_features = ['diet_pref', 'act_level', 'career']
    numeric_features = ['phy_fitness', 'sleep_hrs', 'mindfulness', 'gender', 'daily_avg_steps', 'daily_avg_calories']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])
    
    # Create pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression())
    ])
    
    # Train model
    model.fit(X, y)
    return model

# Streamlit app
def main():
    st.title('Health Prediction App')
    st.write('Enter your health and lifestyle details to predict your health status')
    
    # Load data
    data = load_data()
    
    # Train model
    model = train_model(data)
    
    # Input fields
    with st.form('health_form'):
        phy_fitness = st.slider('Physical Fitness (1-10)', 1, 10, 3)
        diet_pref = st.selectbox('Diet Preference', ['Vegan', 'Vegetarian', 'Non-Vegetarian','Keto','pysechastrian'])
        act_level = st.selectbox('Activity Level', ['Sedentary', 'Lightly Active', 'Active', 'Highly Active'])
        sleep_hrs = st.slider('Hours of Sleep', 0, 12, 7)
        mindfulness = st.slider('Mindfulness Practice (1-10)', 1, 10, 10)
        career = st.selectbox('Career', ['Artist', 'Teacher', 'Freelancer', 'Doctor', 'Business', 'Nurse', 'Lawyer', 'Scientist', 'Engineer', 'Manager'])
        gender = st.radio('Gender', ['Male', 'Female'])
        daily_avg_steps = st.number_input('Daily Average Steps', min_value=0, value=2020)
        daily_avg_calories = st.number_input('Daily Average Calories', min_value=0, value=1831)
        
        submitted = st.form_submit_button('Predict')
        
        if submitted:
            # Prepare input data
            input_data = pd.DataFrame({
                'phy_fitness': [phy_fitness],
                'diet_pref': [diet_pref],
                'act_level': [act_level],
                'sleep_hrs': [sleep_hrs],
                'mindfulness': [mindfulness],
                'career': [career],
                'gender': [1 if gender == 'Male' else 0],
                'daily_avg_steps': [daily_avg_steps],
                'daily_avg_calories': [daily_avg_calories]
            })
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            result = 'Healthy' if prediction == 1 else 'Not Healthy'
            
            st.success(f'Prediction: {result}')

if __name__ == '__main__':
    main()

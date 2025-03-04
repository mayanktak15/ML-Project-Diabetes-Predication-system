import joblib
import streamlit as st
import pandas as pd
import numpy as np
import os

# Set page config
st.set_page_config(page_title="Diabetes Prediction", layout="wide", page_icon="🏥")

# Title and description
st.title('Diabetes Prediction using ML')
st.write('Enter the required information to check diabetes risk')

# Load model with error handling
@st.cache_resource
def load_model():
    try:
        return joblib.load('diabetes')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load data with error handling
@st.cache_data
def load_data():
    try:
        return pd.read_csv('diabetes.csv')
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

diabetes_model = load_model()
df = load_data()

if not df.empty and diabetes_model:
    # Create form for better submission control
    with st.form("prediction_form"):
        st.subheader("Patient Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            Pregnancies = st.selectbox('Number of Pregnancies', sorted(df['Pregnancies'].unique()))
        with col2:
            Glucose = st.selectbox('Glucose Level (mg/dL)', sorted(df['Glucose'].unique()))
        with col3:
            BloodPressure = st.selectbox('Blood Pressure (mm Hg)', sorted(df['BloodPressure'].unique()))

        with col1:
            SkinThickness = st.selectbox('Skin Thickness (mm)', sorted(df['SkinThickness'].unique()))
        with col2:
            Insulin = st.selectbox('Insulin Level (mu U/ml)', sorted(df['Insulin'].unique()))
        with col3:
            BMI = st.selectbox('BMI value (kg/m²)', sorted(df['BMI'].unique()))

        with col1:
            DiabetesPedigreeFunction = st.selectbox('Diabetes Pedigree Function', sorted(df['DiabetesPedigreeFunction'].unique()))
        with col2:
            Age = st.selectbox('Age (years)', sorted(df['Age'].unique()))
        
        # Create a submit button
        submit_button = st.form_submit_button(label="Predict Diabetes Risk")
    
    # Prediction processing
    if submit_button:
        try:
            # Convert input values to appropriate data types
            input_data = np.array([
                float(Pregnancies),
                float(Glucose),
                float(BloodPressure),
                float(SkinThickness),
                float(Insulin),
                float(BMI),
                float(DiabetesPedigreeFunction),
                float(Age)
            ]).reshape(1, -1)  # Correct: -1
            
            # Make prediction
            diab_prediction = diabetes_model.predict(input_data)
            
            # Display result
            st.subheader("Prediction Result")
            if diab_prediction[0] == 1:
                st.error("🔴 The person is diabetic")
                st.info("Please consult with a healthcare professional for proper diagnosis and treatment.")
            else:
                st.success("🟢 The person is not diabetic")
                st.info("Remember to maintain a healthy lifestyle for diabetes prevention.")
                
            # Display input data summary
            st.subheader("Patient Data Summary")
            patient_data = {
                'Parameter': ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 
                             'Insulin', 'BMI', 'Diabetes Pedigree Function', 'Age'],
                'Value': [Pregnancies, Glucose, BloodPressure, SkinThickness, 
                         Insulin, BMI, DiabetesPedigreeFunction, Age]
            }
            st.table(pd.DataFrame(patient_data))
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.info("Please check your input values and try again.")
else:
    st.error("Could not load the model or dataset. Please check the files and try again.")


import streamlit as st
import pandas as pd
import joblib


# Load the trained Random Forest Regressor model
model = joblib.load('app(1).py')

# Load the label encoders
label_encoders = joblib.load('label_encoders.pkl')

st.title('Salary Prediction App')
st.write('Enter the details below to predict the salary:')

# Input fields for features
age = st.slider('Age', 18, 65, 30)
years_of_experience = st.slider('Years of Experience', 0, 40, 5)

# Dropdowns for categorical features, using the inverse mapping from label encoders
gender_options = list(label_encoders['Gender'].classes_)
gender_input = st.selectbox('Gender', gender_options)
gender = label_encoders['Gender'].transform([gender_input])[0]

education_options = list(label_encoders['Education Level'].classes_)
education_level_input = st.selectbox('Education Level', education_options)
education_level = label_encoders['Education Level'].transform([education_level_input])[0]

job_title_options = list(label_encoders['Job Title'].classes_)
job_title_input = st.selectbox('Job Title', job_title_options)
job_title = label_encoders['Job Title'].transform([job_title_input])[0]

# Create a DataFrame for the input features
input_data = pd.DataFrame({
    'Age': [age],
    'Gender': [gender],
    'Education Level': [education_level],
    'Job Title': [job_title],
    'Years of Experience': [years_of_experience]
})

if st.button('Predict Salary'):
    predicted_salary = model.predict(input_data)[0]
    st.success(f'Predicted Salary: ${predicted_salary:,.2f}')

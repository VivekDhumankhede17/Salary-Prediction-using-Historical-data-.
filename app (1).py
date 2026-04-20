import streamlit as st
import pandas as pd
import pickle

# Load the trained model
# Make sure 'best_model.pkl' is in the same directory as app.py or provide the correct path
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit App Title
st.title('Salary Prediction App')
st.write('Enter the details below to predict the salary.')

# Input fields for the features
# Note: The LabelEncoder mappings are assumed to be consistent with training.
# For a robust app, you would save the LabelEncoders as well.

gender_options = {'Male': 1, 'Female': 0} # Based on your Label Encoding
education_options = {
    "Bachelor's": 0,
    "Master's": 1,
    "PhD": 2
} # Based on your Label Encoding

age = st.slider('Age', 18, 65, 30)
gender = st.selectbox('Gender', list(gender_options.keys()))
education_level = st.selectbox('Education Level', list(education_options.keys()))
years_of_experience = st.slider('Years of Experience', 0, 40, 5)

# Job Title is more complex due to many categories. For simplicity, we'll use a placeholder or assume fixed values for now.
# In a real-world scenario, you would have a more sophisticated way to handle many categories.
# For this demo, let's assume 'Software Engineer' is the most common or a default.
# The actual encoded value for 'Software Engineer' was 159. Adjust if needed.
# For a full application, you'd need the saved label_encoder for 'Job Title'.
job_title = st.number_input('Job Title Encoded (e.g., 159 for Software Engineer)', min_value=0, max_value=200, value=159)

# Preprocess input
gender_encoded = gender_options[gender]
education_encoded = education_options[education_level]

# Create a DataFrame for prediction
input_data = pd.DataFrame([[age, gender_encoded, education_encoded, job_title, years_of_experience]],
                            columns=['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience'])

# Predict button
if st.button('Predict Salary'):
    prediction = model.predict(input_data)[0]
    st.success(f'Predicted Salary: ${prediction:,.2f}')

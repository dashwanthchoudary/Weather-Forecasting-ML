import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Load the saved model
model_file = 'polynomial_regression_model.joblib'
model = joblib.load(model_file)

# Define a function to predict temperature
def predict_temperature(features):
    # Transform features for Polynomial Regression
    degree = 2  # Ensure this matches the degree used during training
    poly = PolynomialFeatures(degree)
    features_poly = poly.fit_transform(features)
    
    # Predict using the loaded model
    temperature_prediction = model.predict(features_poly)
    return temperature_prediction

# Streamlit app begins
st.title('Temperature Prediction App')

# Sidebar for input features
st.sidebar.header('Input Features')

# Example input fields (replace with actual user input or sliders)
default_input = {
    'Apparent Temperature (C)': 20.0,
    'Humidity': 0.5,
    'Wind Speed (km/h)': 10.0,
    'Wind Bearing (degrees)': 180.0,
    'Visibility (km)': 10.0,
    'Pressure (millibars)': 1000.0
}

# Collect user input through sidebar
user_input = {}
for key, value in default_input.items():
    user_input[key] = st.sidebar.number_input(key, value=value)

# Function to predict and display temperature on button click
if st.sidebar.button('Predict Temperature'):
    # Convert user input into DataFrame
    input_features = pd.DataFrame([user_input])

    # Make prediction
    predicted_temperature = predict_temperature(input_features)

    # Display prediction
    st.write('## Predicted Temperature')
    st.write(f'The predicted temperature is {predicted_temperature[0]:.2f} °C')

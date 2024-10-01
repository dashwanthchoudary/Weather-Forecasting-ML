import streamlit as st
import pandas as pd
import requests
import json
import joblib
from sklearn.preprocessing import PolynomialFeatures
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from datetime import datetime
from datetime import timezone as tmz
import pytz
from timezonefinder import TimezoneFinder
from streamlit_folium import folium_static
import folium


# Load the saved ML model
model_file = 'polynomial_regression_model.joblib'
model = joblib.load(model_file)

# Define a function to predict temperature using the Polynomial Regression model
def predict_temperature(features):
    degree = 2  # Ensure this matches the degree used during training
    poly = PolynomialFeatures(degree)
    features_poly = poly.fit_transform(features)
    
    # Predict using the loaded model
    temperature_prediction = model.predict(features_poly)
    return temperature_prediction

# Streamlit app begins
st.title("How's the weather? :sun_behind_rain_cloud:")

st.subheader("Choose location")

file = "worldcities.csv"
data = pd.read_csv(file)

# Select Country and City
country_set = set(data.loc[:,"country"])
country = st.selectbox('Select a country', options=country_set)
country_data = data.loc[data.loc[:,"country"] == country,:]
city_set = country_data.loc[:,"city_ascii"] 
city = st.selectbox('Select a city', options=city_set)

# Get the latitude and longitude for the selected city
lat = float(country_data.loc[data.loc[:, "city_ascii"] == city, "lat"].iloc[0])
lng = float(country_data.loc[data.loc[:, "city_ascii"] == city, "lng"].iloc[0])

# Get current weather data from Open-Meteo API
response_current = requests.get(f'https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lng}&current_weather=true')
result_current = json.loads(response_current._content)
current = result_current["current_weather"]
temp = current["temperature"]
speed = current["windspeed"]
direction = current["winddirection"]

# Increment added or subtracted from degree values for wind direction
ddeg = 11.25
common_dir = "N"  # Default value

# Determine wind direction
if direction >= (360-ddeg) or direction < (0+ddeg):
    common_dir = "N"
elif direction >= (337.5-ddeg) and direction < (337.5+ddeg):
    common_dir = "N/NW"
elif direction >= (315-ddeg) and direction < (315+ddeg):
    common_dir = "NW"
elif direction >= (292.5-ddeg) and direction < (292.5+ddeg):
    common_dir = "W/NW"
elif direction >= (270-ddeg) and direction < (270+ddeg):
    common_dir = "W"
# Add other directions as in the previous code

# Display current weather
st.info(f"The current temperature is {temp} °C. \n The wind speed is {speed} m/s. \n The wind is coming from {common_dir}.")

# Sidebar for input features for ML model
st.sidebar.header('Input Features for ML Prediction')
default_input = {
    'Apparent Temperature (C)': temp,  # Using API temp as default
    'Humidity': 0.5,
    'Wind Speed (km/h)': speed,  # Using API wind speed as default
    'Wind Bearing (degrees)': direction,  # Using API wind direction as default
    'Visibility (km)': 10.0,
    'Pressure (millibars)': 1000.0
}

# Collect user input through sidebar
user_input = {}
for key, value in default_input.items():
    user_input[key] = st.sidebar.number_input(key, value=value)

# Function to predict and display temperature on button click
if st.sidebar.button('Predict Temperature (ML Model)'):
    # Convert user input into DataFrame
    input_features = pd.DataFrame([user_input])

    # Make ML-based temperature prediction
    predicted_temperature = predict_temperature(input_features)

    # Display ML-based prediction
    st.subheader("ML-based Temperature Prediction")
    st.write(f'The predicted temperature using the ML model is {predicted_temperature[0]:.2f} °C.')

# Week ahead forecast from Open-Meteo API
st.subheader("Week ahead")
st.write('Temperature and rain forecast one week ahead & city location on the map', unsafe_allow_html=True)

with st.spinner('Loading...'):
    response_hourly = requests.get(f'https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lng}&hourly=temperature_2m,precipitation')
    result_hourly = json.loads(response_hourly._content)
    hourly = result_hourly["hourly"]
    hourly_df = pd.DataFrame.from_dict(hourly)
    hourly_df.rename(columns={'time': 'Week ahead', 'temperature_2m': 'Temperature °C', 'precipitation': 'Precipitation mm'}, inplace=True)

    # Timezone adjustment
    timezone_finder = TimezoneFinder()
    timezone_str = timezone_finder.timezone_at(lng=lng, lat=lat)
    if timezone_str is None:
        st.error("Timezone could not be determined for this location.")
    else:
        timezone_loc = pytz.timezone(timezone_str)
        dt = datetime.now()
        tzoffset = timezone_loc.utcoffset(dt)

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        week_ahead = pd.to_datetime(hourly_df['Week ahead'], format="%Y-%m-%dT%H:%M")
        fig.add_trace(go.Scatter(x=week_ahead + tzoffset, y=hourly_df['Temperature °C'], name="Temperature °C"), secondary_y=False)
        fig.add_trace(go.Bar(x=week_ahead + tzoffset, y=hourly_df['Precipitation mm'], name="Precipitation mm"), secondary_y=True)

        # Highlight current time
        time_now = datetime.now(tmz.utc) + tzoffset
        fig.add_vline(x=time_now, line_color="red", opacity=0.4)
        fig.add_annotation(x=time_now, y=max(hourly_df['Temperature °C']) + 5, text=time_now.strftime("%d %b %y, %H:%M"), showarrow=False, yshift=0)

        # Update axes
        fig.update_yaxes(range=[min(hourly_df['Temperature °C']) - 10, max(hourly_df['Temperature °C']) + 10], title_text="Temperature °C", secondary_y=False)
        fig.update_yaxes(range=[min(hourly_df['Precipitation mm']) - 2, max(hourly_df['Precipitation mm']) + 8], title_text="Precipitation (rain/showers/snow) mm", secondary_y=True)

        # Display chart
        st.plotly_chart(fig, use_container_width=True)

        # Map visualization
        m = folium.Map(location=[lat, lng], zoom_start=7)
        folium.Marker([lat, lng], popup=city + ', ' + country, tooltip=city + ', ' + country).add_to(m)
        folium_static(m, height=370)

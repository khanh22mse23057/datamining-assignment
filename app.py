import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import datetime as dt
import category_encoders as ce
import joblib
import warnings
warnings.filterwarnings('ignore')
from xgboost import XGBRegressor
import locale

# Load your CSV data (replace 'your_data.csv' with your actual CSV file)
@st.cache  # Cache the data for improved performance
def load_data():
    data = pd.read_csv('car_price_cleaned.csv')
    return data

# Function to extract manufacturer and model from a given car name
def extract_manufacturer_and_model(car_name):
    # Implement your logic to extract manufacturer and model
    # For example, split the car_name based on a known separator
    parts = car_name.split('-')
    if len(parts) >= 2:
        manufacturer = parts[0].strip()
        model = parts[1].strip()
        return manufacturer, model
    else:
        return None, None
    
# Custom HTML code with author information
custom_html = """
<div style="background-color: #4b5a7a; padding: 1.5px">
    <h1 style="color: #48bf08; text-align: center;">Car Price Prediction Project</h1>
    <h4 style="color: #a80f63; text-align: center;">Team Members:</h4>
    <ul style="list-style-type: none; padding-left: 0; text-align: center;">
        <li>Hoàng Tuấn Anh</li>
        <li>Nguyễn Văn Vũ</li>
        <li>Phạm Nguyễn Phú Khánh</li>
        <li>Nguyễn Minh Hiếu</li>
    </ul>
</div><br>
"""
st.markdown(custom_html, unsafe_allow_html=True)

# Load the CSV data
manufacturer_data  = load_data()

st.write("\n\n"*2)

# filename = 'carPricePredictorModel.pkl'
# price_model_ = pickle.load(open(filename, 'rb'))
# Save the model
price_model_=joblib.load('carPricePredictorModel.pkl')

with st.sidebar:
    st.subheader('Car Specs to Predict Price')

manufacturer = st.sidebar.selectbox("Brand", (manufacturer_data['manufacturer'].unique()))

filtered_data = manufacturer_data[manufacturer_data['manufacturer'] == manufacturer]
model = st.sidebar.selectbox("Model Selection", (filtered_data['model'].unique()))

engine_turbo = st.checkbox('Has Turbo')

doors = st.sidebar.selectbox("Door", ("2-3", "4-5", ">5"))
wheel = st.sidebar.selectbox("Wheel Type", ("Left", "Right"))
levy = st.sidebar.number_input("Levy:", min_value=0, max_value=3000, value=1000, step=100)
cylinders = st.sidebar.number_input("Cylinders:", min_value=1, max_value=16, value=4, step=4)
fuel_type = st.sidebar.selectbox("Fuel Type", ('Hybrid', 'Petrol', 'Diesel', 'CNG', 'Plug-in Hybrid', 'LPG', 'Hydrogen'))
engine_volume = st.sidebar.number_input("Engine Volume:", min_value=0.9, max_value=3.6, value=3.0, step=0.1)
engine_volume = float(engine_volume)
mileage = st.sidebar.number_input("Mileage/Km:", min_value=0, max_value=36200, value=10000, step=5000)
airbags = st.sidebar.number_input("Airbags:", min_value=2, max_value=13, value=4, step=1)
gear_box_type = st.sidebar.radio("Gearing Type", ('Automatic', 'Tiptronic', 'Variator', 'Manual'))
drive_wheels = st.sidebar.radio("Drive Wheels", ('4x4', 'Front', 'Rear'))

age = st.sidebar.number_input("Age:", min_value=3, max_value=80, value=5, step=1)
category = st.sidebar.radio("Car Type", ('Jeep', 'Hatchback', 'Sedan', 'Microbus', 'Goods wagon',
       'Universal', 'Coupe', 'Minivan', 'Cabriolet', 'Limousine',
       'Pickup'))

    
my_dict = {
    'doors': doors,
    'wheel': wheel,
    'levy': levy,
    'engine_volume': engine_volume,
    'mileage': mileage,
    'cylinders': cylinders,
    'airbags': airbags,
    'model': 'Corolla',
    'category': category,
    'leather_interior': 'Yes',
    'fuel_type': fuel_type,
    'gear_box_type': gear_box_type,
    'drive_wheels': drive_wheels,
    'engine_turbo': engine_turbo,
    'age': age,
    'manufacturer': manufacturer
}

# df = pd.DataFrame.from_dict([my_dict])
_data = pd.DataFrame([my_dict])

columns = ['levy', 'manufacturer', 'model', 'category', 'leather_interior',
       'fuel_type', 'engine_volume', 'mileage', 'cylinders', 'gear_box_type',
       'drive_wheels', 'doors', 'wheel', 'airbags', 'engine_turbo', 'age']

cols = {
    "levy": "Levy",
    "manufacturer": "Manufacturer",
    "model": "  Model",
    "category": "Category",
    "leather_interior": "Leather interior",
    "fuel_type": "Fuel type",
    "engine_volume": "Engine volume",
    "mileage": " Mileage(Km)",
    "cylinders": "Cylinders",
    "gear_box_type": "Gear box type",
    "drive_wheels": "Drive wheels",
    "doors": " Doors",
    "wheel": "Gearing Type",
    "airbags": "Airbags",
    "engine_turbo": "Engine Turbo",
    "age": "Age"

}

df_show = _data.copy()
df_show.rename(columns=cols, inplace=True)
st.write("Selected Specs: \n")
st.table(df_show)

# Set the locale to your desired currency format
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')  # For US currency format

if st.button("Predict"):
    pred = price_model_.predict(_data)
    col1, col2 = st.columns(2)
    col1.write("The approximate cost of a car is:")
    cost = float(pred[0])
    formatted_currency = locale.currency(cost, grouping=True)
    col2.write(formatted_currency)

st.write("\n\n")


# import streamlit as st  # type: ignore
# import pandas as pd # type: ignore
# from sklearn.ensemble import RandomForestClassifier # type: ignore

# st.title("Air Pollution Level Detection")

# df = pd.read_csv("air_pollution_dataset_75000.csv")

# X = df.drop("AirQualityLevel", axis=1)
# y = df["AirQualityLevel"]

# model = RandomForestClassifier(n_estimators=120, random_state=42)
# model.fit(X, y)

# st.header("Enter Pollution Parameters")

# pm25 = st.number_input("PM2.5")
# pm10 = st.number_input("PM10")
# no2 = st.number_input("NO2")
# so2 = st.number_input("SO2")
# co = st.number_input("CO")
# o3 = st.number_input("O3")
# temp = st.number_input("Temperature")
# humidity = st.number_input("Humidity")
# wind = st.number_input("Wind Speed")

# if st.button("Predict Air Quality"):
#     input_data = pd.DataFrame([[pm25, pm10, no2, so2, co, o3, temp, humidity, wind]],
#         columns=X.columns)

#     prediction = model.predict(input_data)[0]

#     st.success(f"Predicted Air Quality Level: {prediction}")


import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title("Air Pollution Level Detection")

# Load dataset
df = pd.read_csv("dataset/air_pollution_dataset_75000.csv")

X = df.drop("AirQualityLevel", axis=1)
y = df["AirQualityLevel"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Inputs
pm25 = st.number_input("PM2.5")
pm10 = st.number_input("PM10")
no2 = st.number_input("NO2")
so2 = st.number_input("SO2")
co = st.number_input("CO")
o3 = st.number_input("O3")
temp = st.number_input("Temperature")
humidity = st.number_input("Humidity")
wind = st.number_input("Wind Speed")

if st.button("Predict"):

    data = pd.DataFrame([[pm25,pm10,no2,so2,co,o3,temp,humidity,wind]],
    columns=X.columns)

    prediction = model.predict(data)

    st.success(f"Predicted Air Quality Level: {prediction[0]}")

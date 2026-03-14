
# # import streamlit as st  # type: ignore
# # import pandas as pd # type: ignore
# # from sklearn.ensemble import RandomForestClassifier # type: ignore

# # st.title("Air Pollution Level Detection")

# # df = pd.read_csv("air_pollution_dataset_75000.csv")

# # X = df.drop("AirQualityLevel", axis=1)
# # y = df["AirQualityLevel"]

# # model = RandomForestClassifier(n_estimators=120, random_state=42)
# # model.fit(X, y)

# # st.header("Enter Pollution Parameters")

# # pm25 = st.number_input("PM2.5")
# # pm10 = st.number_input("PM10")
# # no2 = st.number_input("NO2")
# # so2 = st.number_input("SO2")
# # co = st.number_input("CO")
# # o3 = st.number_input("O3")
# # temp = st.number_input("Temperature")
# # humidity = st.number_input("Humidity")
# # wind = st.number_input("Wind Speed")

# # if st.button("Predict Air Quality"):
# #     input_data = pd.DataFrame([[pm25, pm10, no2, so2, co, o3, temp, humidity, wind]],
# #         columns=X.columns)

# #     prediction = model.predict(input_data)[0]

# #     st.success(f"Predicted Air Quality Level: {prediction}")


# import streamlit as st
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier

# st.title("Air Pollution Level Detection")

# # Load dataset
# df = pd.read_csv("dataset/air_pollution_dataset_75000.csv")

# X = df.drop("AirQualityLevel", axis=1)
# y = df["AirQualityLevel"]

# # Train model
# model = RandomForestClassifier()
# model.fit(X, y)

# # Inputs
# pm25 = st.number_input("PM2.5")
# pm10 = st.number_input("PM10")
# no2 = st.number_input("NO2")
# so2 = st.number_input("SO2")
# co = st.number_input("CO")
# o3 = st.number_input("O3")
# temp = st.number_input("Temperature")
# humidity = st.number_input("Humidity")
# wind = st.number_input("Wind Speed")

# if st.button("Predict"):

#     data = pd.DataFrame([[pm25,pm10,no2,so2,co,o3,temp,humidity,wind]],
#     columns=X.columns)

#     prediction = model.predict(data)

#     st.success(f"Predicted Air Quality Level: {prediction[0]}")


import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Air Pollution Detection", layout="wide")

st.title("🌍 Air Pollution Level Detection System")

st.write(
"This application predicts air quality levels using a Machine Learning model "
"trained on pollution and environmental parameters."
)

# Load dataset
df = pd.read_csv("air_pollution_dataset_75000.csv")

# Load trained model
model = joblib.load("trained_model.pkl")

# Sidebar
st.sidebar.header("About")
st.sidebar.write(
"This project predicts air pollution levels using a Random Forest Machine Learning model."
)

st.sidebar.write("Developed using Python, Scikit-Learn, and Streamlit.")

# Show dataset preview
if st.checkbox("Show Dataset"):
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

# Show dataset statistics
if st.checkbox("Show Dataset Statistics"):
    st.subheader("Statistical Summary")
    st.write(df.describe())

# Visualization
st.subheader("Data Visualization")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.histplot(df["PM2_5"], kde=True, ax=ax)
    ax.set_title("PM2.5 Distribution")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.histplot(df["PM10"], kde=True, ax=ax)
    ax.set_title("PM10 Distribution")
    st.pyplot(fig)

# User input section
st.subheader("Enter Pollution Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    pm25 = st.number_input("PM2.5", 0.0, 500.0, 50.0)
    pm10 = st.number_input("PM10", 0.0, 500.0, 80.0)
    no2 = st.number_input("NO2", 0.0, 200.0, 30.0)

with col2:
    so2 = st.number_input("SO2", 0.0, 200.0, 20.0)
    co = st.number_input("CO", 0.0, 10.0, 0.5)
    o3 = st.number_input("O3", 0.0, 200.0, 40.0)

with col3:
    temp = st.number_input("Temperature (°C)", -10.0, 50.0, 25.0)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)
    wind = st.number_input("Wind Speed (km/h)", 0.0, 100.0, 10.0)

# Prediction
if st.button("Predict Air Quality"):

    input_data = pd.DataFrame(
        [[pm25, pm10, no2, so2, co, o3, temp, humidity, wind]],
        columns=["PM2_5","PM10","NO2","SO2","CO","O3","Temperature","Humidity","WindSpeed"]
    )

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)

    st.subheader("Prediction Result")

    if prediction == 0:
        st.success("🟢 Good Air Quality")

    elif prediction == 1:
        st.warning("🟡 Moderate Air Quality")

    else:
        st.error("🔴 Poor Air Quality")

    st.write("Prediction Probabilities:")
    st.write(probability)

# Correlation heatmap
if st.checkbox("Show Correlation Heatmap"):
    st.subheader("Feature Correlation")

    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

st.markdown("---")
st.caption("Machine Learning Project - Air Pollution Prediction")

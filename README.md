# 🚴 Bike Rental Demand Prediction

A Machine Learning web application that predicts hourly bike rental demand based on time, weather, and environmental conditions.

The project uses an optimized **XGBoost Regression model** and provides an interactive **Streamlit dashboard** for making predictions and exploring the bike rental dataset.

---

## 📌 Project Overview

Bike rental demand changes based on several factors such as:

- Hour of the day
- Month
- Weekday
- Working day
- Holiday
- Temperature
- Feeling temperature
- Humidity
- Windspeed
- Season
- Weather situation

This project uses historical bike rental data to train a machine learning model and predict the expected number of bike rentals for a selected hour.

---

## 🎯 Objective

The main objective of this project is to:

- Predict hourly bike rental demand.
- Identify important factors affecting bike rentals.
- Build an easy-to-use machine learning application.
- Visualize patterns and trends in bike rental demand.
- Deploy the prediction model through a Streamlit web application.

---

## 🤖 Machine Learning Model

The project uses **XGBoost Regression** for predicting bike rental demand.

### Model Used

**XGBoost Regressor**

XGBoost was selected because it performs well on structured/tabular data and can capture nonlinear relationships between input features and rental demand.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib
- Streamlit
- Matplotlib / Data Visualization
- Jupyter Notebook
- GitHub

---

## 📊 Features

### Time Features

- Year
- Month
- Hour
- Weekday

### Day Features

- Holiday
- Working Day
- Weekend

### Weather Features

- Temperature
- Feeling Temperature
- Humidity
- Windspeed
- Weather Situation

### Additional Features

- Peak Hour
- Hour Bucket
- Temperature Difference
- Season

---

## 📈 Data Visualizations

The Streamlit application includes interactive visualizations such as:

- ⏰ Average Bike Rental Demand by Hour
- 📅 Monthly Bike Rental Demand
- 🍂 Season-wise Rental Demand
- 💼 Working Day vs Non-Working Day
- 🌦️ Weather Situation vs Rental Demand
- 🌡️ Temperature vs Rental Demand
- 💧 Humidity vs Rental Demand
- 💨 Windspeed vs Rental Demand
- 🔥 Peak Hour vs Normal Hour

---

## 🚀 Streamlit Application

The application allows users to select different input conditions from the sidebar and generate a bike rental demand prediction.

### Example Inputs

```text
Year              : 2012
Month             : 5
Hour              : 08:00
Day of Week       : 3
Holiday           : No
Working Day       : Working Day
Temperature       : 0.37
Feeling Temperature: 0.40
Humidity          : 0.50
Windspeed         : 0.20
Season            : Summer
Weather Situation : Clear

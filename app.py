
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")

# Merge datasets
exercise_df = exercise.merge(calories, on="User_ID")

# Selecting features and target
target = exercise_df['Calories']
feature = exercise_df[['Heart_Rate']]

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.11, random_state=101)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "calories_model.pkl")

# Streamlit UI
st.title("Calories Prediction App")

# Input from user
heart_rate = st.number_input("Enter Heart Rate:", min_value=50, max_value=200, value=70)

# Load model & predict
model = joblib.load("calories_model.pkl")
prediction = model.predict([[heart_rate]])

st.write(f"Predicted Calories Burned: {prediction[0]:.2f}")

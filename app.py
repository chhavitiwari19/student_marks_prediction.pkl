import streamlit as st
import pickle
import numpy as np
# load the trained ml model
with open("model.pkl","rb") as file:
  model = pickle.load(file)
# website title
st.title("student marks prediction")
st.write("predict student marks using study hours and attendance.")
# user inputs
Study_hours = st.number_input(
  "study hours per day",
  min_value=0.0,
  max_value=24.0,
  value=20.0,
  step=0.5 )
attendance = st.number_input(
  "Attendance (%)",
  min_value=0.0,
  max_value=100.0,
  value=75.0,
  step=1.0 )
# predicted marks
if st.button("Predict marks"):
  features = np.array([[Study_hours, attendance]])
  prediction = model.predict(features)
  st.success(f"predicted marks:{prediction[0]:.2f}")

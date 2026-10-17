import pandas as pd
import streamlit as st
import pickle

with open("student_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("AI Student Performance Predictor")

st.write("Enter the student's study details below.")

study_hours = st.slider(
    "Daily study hours",
    min_value=0,
    max_value=12,
    value=4
)

attendance = st.slider(
    "Attendance percentage",
    min_value=0,
    max_value=100,
    value=80
)

previous_score = st.slider(
    "Previous exam score",
    min_value=0,
    max_value=100,
    value=65
)

st.write("Study hours:", study_hours)
st.write("Attendance:", attendance, "%")
st.write("Previous score:", previous_score)
if st.button("Analyze Student"):
    input_data = pd.DataFrame(
        [[study_hours, attendance, previous_score]],
        columns=["study_hours", "attendance", "previous_score"]
    )

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")
    st.success(f"Predicted performance category: {prediction}")
    st.caption(
        "Educational demo only. This model uses a tiny, "
        "artificial sample dataset and is not reliable for "
        "real student assessment."
    )
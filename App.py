import streamlit as st
import pandas as pd
import joblib

model = joblib.load('fighter_performance_model.pkl')

st.title(' MMA Fighter High Performance Predictor')

st.markdown("Fill in the fighter’s stats below to predict if they are a high performer.")


if st.button(" Reset All Inputs"):
    st.experimental_rerun()

with st.expander(" Input Feature Explanation"):
    st.markdown("""
    - **Height, Weight, Reach**: Fighter's physical stats in cm/kg.
    - **Age**: Fighter’s current age (18–55 range).
    - **Striking Metrics**:
        - `Significant Strikes Landed/Min`: Avg punches/kicks landed.
        - `Striking Accuracy (%)`: Percent of strikes that hit the opponent.
        - `Strikes Absorbed/Min`: Avg hits received per minute.
        - `Strike Defense (%)`: Percent of opponent's strikes avoided.
    - **Grappling Metrics**:
        - `Takedowns/15min`: Number of takedowns landed.
        - `Takedown Accuracy (%)`: Success rate of takedown attempts.
        - `Takedown Defense (%)`: Defense rate against takedowns.
        - `Submissions/15min`: Number of submission attempts.
    """)


height = st.number_input('Height (cm)', min_value=150.0, max_value=220.0, key='height')
weight = st.number_input('Weight (kg)', min_value=40.0, max_value=150.0, key='weight')
reach = st.number_input('Reach (cm)', min_value=150.0, max_value=220.0, key='reach')
age = st.number_input('Age', min_value=18, max_value=55, key='age')

strike_rate = st.number_input("Significant Strikes Landed/Min", 0.0, 10.0, key='strike_rate')
strike_acc = st.number_input("Significant Striking Accuracy (%)", 0.0, 100.0, key='strike_acc')
strike_absorb = st.number_input("Strikes Absorbed/Min", 0.0, 10.0, key='strike_absorb')
strike_def = st.number_input("Strike Defense (%)", 0.0, 100.0, key='strike_def')

takedown_rate = st.number_input("Takedowns/15min", 0.0, 15.0, key='takedown_rate')
takedown_acc = st.number_input("Takedown Accuracy (%)", 0.0, 100.0, key='takedown_acc')
takedown_def = st.number_input("Takedown Defense (%)", 0.0, 100.0, key='takedown_def')
submissions = st.number_input("Submissions/15min", 0.0, 10.0, key='submissions')


if st.button(' Predict'):
    input_data = pd.DataFrame([[
        height, weight, reach, age, 
        strike_rate, strike_acc, strike_absorb, strike_def,
        takedown_rate, takedown_acc, takedown_def, submissions
    ]], columns=[
        'height_cm', 'weight_in_kg', 'reach_in_cm', 'age',
        'significant_strikes_landed_per_minute', 'significant_striking_accuracy',
        'significant_strikes_absorbed_per_minute', 'significant_strike_defence',
        'average_takedowns_landed_per_15_minutes', 'takedown_accuracy',
        'takedown_defense', 'average_submissions_attempted_per_15_minutes'
    ])

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f" High Performer predicted with confidence **{prob*100:.2f}%**")
    else:
        st.warning(f" Likely NOT a High Performer (confidence **{prob*100:.2f}%**)")


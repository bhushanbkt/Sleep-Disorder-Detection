import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('model (1).pkl')

def preprocess_input_data(gender, age, occupation, sleep_duration, quality_of_sleep,
                          physical_activity_level, stress_level, bmi_category,
                          blood_pressure, heart_rate, daily_steps):
    # Convert categorical features to numerical values
    gender = 0 if gender == "Male" else 1

    # Use occupation categories provided
    occupation_mapping = {
        'Software Engineer': 0,
        'Doctor': 1,
        'Sales Representative': 2,
        'Teacher': 3,
        'Nurse': 4,
        'Engineer': 5,
        'Accountant': 6,
        'Scientist': 7,
        'Lawyer': 8,
        'Salesperson': 9,
        'Manager': 10
    }
    occupation = occupation_mapping.get(occupation, 0)

    # Map BMI categories to numerical values
    bmi_category_mapping = {"Underweight": 0, "Normal": 1, "Overweight": 2, "Obese": 3}
    bmi_category = bmi_category_mapping.get(bmi_category, 0)

    # Map blood pressure categories to numerical values
    blood_pressure_mapping = {"Low": 0, "Normal": 1, "High": 2}
    blood_pressure = blood_pressure_mapping.get(blood_pressure, 0)

    # Create a dataframe with user inputs
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Age': [age],
        'Occupation': [occupation],
        'Sleep Duration': [sleep_duration],
        'Quality of Sleep': [quality_of_sleep],
        'Physical Activity Level': [physical_activity_level],
        'Stress Level': [stress_level],
        'BMI Category': [bmi_category],
        'Blood Pressure': [blood_pressure],
        'Heart Rate': [heart_rate],
        'Daily Steps': [daily_steps]
    })

    return input_data

def main():
    st.title("Sleep Disorder Prediction App")

    # Collect input features from the user
    gender = st.selectbox("Select Gender", ["Male", "Female"])
    age = st.slider("Select Age", min_value=18, max_value=100, value=30)

    # Use occupation categories provided
    occupation_categories = ['Software Engineer', 'Doctor', 'Sales Representative', 'Teacher',
                             'Nurse', 'Engineer', 'Accountant', 'Scientist', 'Lawyer',
                             'Salesperson', 'Manager']
    occupation = st.selectbox("Select Occupation", occupation_categories)

    sleep_duration = st.slider("Select Sleep Duration", min_value=0.0, max_value=24.0, value=7.0)
    quality_of_sleep = st.slider("Select Quality of Sleep", min_value=0, max_value=10, value=5)
    physical_activity_level = st.slider("Select Physical Activity Level", min_value=0, max_value=10, value=5)
    stress_level = st.slider("Select Stress Level", min_value=0, max_value=10, value=5)
    bmi_category = st.selectbox("Select BMI Category", ["Underweight", "Normal", "Overweight", "Obese"])
    blood_pressure = st.selectbox("Select Blood Pressure", ["Low", "Normal", "High"])
    heart_rate = st.slider("Select Heart Rate", min_value=40, max_value=200, value=80)
    daily_steps = st.slider("Select Daily Steps", min_value=0, max_value=20000, value=5000)

    # Preprocess input data
    input_data = preprocess_input_data(gender, age, occupation, sleep_duration, quality_of_sleep,
                                       physical_activity_level, stress_level, bmi_category,
                                       blood_pressure, heart_rate, daily_steps)

    # Make predictions
    predictions = model.predict_proba(input_data)

    # Map class indices to corresponding labels
    class_labels = {0: "NO Sleep Disorder", 1: "Sleep Apnea", 2: "Insomnia"}
    mapped_predictions = [class_labels[idx] for idx in range(len(class_labels))]

    # Display the raw predictions with class labels
    st.write("Predictions:", dict(zip(mapped_predictions, predictions[0])))

    # Get the class with the highest probability
    predicted_class = predictions.argmax(axis=1)[0]


    # Button to show additional information if needed
    if st.button("Show More Info"):
        st.subheader("Sleep Apnea:")
        st.write(
            "Sleep apnea is a sleep disorder characterized by pauses in breathing or periods of shallow breathing during sleep. These pauses can last for a few seconds to minutes and may occur multiple times during the night.")
        st.write("Types:")
        st.write(
            "- Obstructive Sleep Apnea (OSA): This is the more common type and is caused by the relaxation of throat muscles, leading to a blockage of the airway.")
        st.write(
            "- Central Sleep Apnea (CSA): In this type, the brain fails to send the right signals to the muscles that control breathing.")
        st.write("Symptoms:")
        st.write("- Loud snoring.")
        st.write("- Episodes of breathing cessation observed by others.")
        st.write("- Gasping or choking during sleep.")
        st.write("- Excessive daytime sleepiness.")
        st.write("- Difficulty concentrating.")
        st.write("- Morning headaches.")
        st.write("Risk Factors:")
        st.write("- Obesity.")
        st.write("- Neck circumference (a thicker neck may narrow the airway).")
        st.write("- Being male.")
        st.write("- Family history.")
        st.write("- Age (older adults are at higher risk).")
        st.write("Treatment:")
        st.write("- Lifestyle changes (weight loss, positional therapy).")
        st.write("- Continuous Positive Airway Pressure (CPAP) therapy.")
        st.write("- Oral appliances.")
        st.write("- Surgery (in severe cases).")

        st.subheader("Insomnia:")
        st.write(
            "Insomnia is a sleep disorder characterized by difficulty falling asleep, staying asleep, or experiencing non-restorative sleep. It can be acute (short-term) or chronic (long-term).")
        st.write("Types:")
        st.write("- Transient Insomnia: Lasts for a few nights or weeks.")
        st.write("- Acute Insomnia: Occurs sporadically but often.")
        st.write("- Chronic Insomnia: Lasts for at least three nights a week for three months or more.")
        st.write("Symptoms:")
        st.write("- Difficulty falling asleep.")
        st.write("- Waking up frequently during the night.")
        st.write("- Difficulty returning to sleep.")
        st.write("- Waking up too early in the morning.")
        st.write("- Non-restorative sleep.")
        st.write("- Daytime sleepiness.")
        st.write("- Irritability or anxiety related to sleep.")
        st.write("Risk Factors:")
        st.write("- Stress and anxiety.")
        st.write("- Depression.")
        st.write("- Chronic medical conditions.")
        st.write("- Changes in sleep environment or routine.")
        st.write("- Certain medications.")
        st.write("- Substance abuse.")
        st.write("Treatment:")
        st.write("- Cognitive-behavioral therapy for insomnia (CBT-I).")
        st.write("- Sleep hygiene practices.")
        st.write("- Medications (prescription or over-the-counter).")
        st.write("- Addressing underlying health issues.")
        st.write("- Stress management and relaxation techniques.")


if __name__ == '__main__':
    main()

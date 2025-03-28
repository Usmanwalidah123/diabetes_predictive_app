import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes.csv")
    return df

# 2. Train Model
@st.cache_data
def train_model(df):
    # Assume the columns in your dataset are:
    # 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
    # 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate model accuracy on test data
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, acc

def main():
    st.title("Diabetes Predictive App")
    st.write("A simple web app to predict if someone has diabetes based on certain health parameters.")

    # Load the data
    df = load_data()

    # Train the model
    model, accuracy = train_model(df)

    st.write(f"**Model Accuracy:** {accuracy:.2f}")

    # Create input widgets for user
    st.subheader("Enter Patient Data:")
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("BloodPressure", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("SkinThickness", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulin", min_value=0, max_value=1000, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
    dpf = st.number_input("DiabetesPedigreeFunction", min_value=0.0, max_value=2.5, value=0.5, step=0.01)
    age = st.number_input("Age", min_value=0, max_value=120, value=30)

    # When user clicks 'Predict'
    if st.button("Predict"):
        # Create a NumPy array from the user inputs
        input_data = np.array([[pregnancies, glucose, blood_pressure, 
                                skin_thickness, insulin, bmi, dpf, age]])

        prediction = model.predict(input_data)

        # Show result
        if prediction[0] == 1:
            st.error("The model predicts that this patient has diabetes.")
        else:
            st.success("The model predicts that this patient does NOT have diabetes.")

if __name__ == "__main__":
    main()

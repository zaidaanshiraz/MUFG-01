import streamlit as st
import pickle
import numpy as np

# Load trained model
with open("lr_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Logistic Regression Iris Classifier 🌸")

st.write("Enter flower features to predict species:")

# Input fields (Iris dataset has 4 features)
sepal_length = st.number_input("Sepal Length (cm)", value=5.1)
sepal_width  = st.number_input("Sepal Width (cm)", value=3.5)
petal_length = st.number_input("Petal Length (cm)", value=1.4)
petal_width  = st.number_input("Petal Width (cm)", value=0.2)

# Predict button
if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]

    target_names = ["Setosa", "Versicolor", "Virginica"]
    st.success(f"🌼 Predicted Species: {target_names[prediction]}")
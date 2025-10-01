import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle
import os

# --- Constants ---
DATASET_PATH = 'manufacturing_dataset_1000_samples.csv'
MODEL_PATH = 'linear_regression_model.pkl'

# --- Model Training ---
def train_model():
    """
    Trains a linear regression model on the manufacturing dataset and saves it.
    """
    if not os.path.exists(DATASET_PATH):
        st.error(f"Dataset not found at {DATASET_PATH}. Please make sure it's in the same directory.")
        return None, None

    df = pd.read_csv(DATASET_PATH)

    # Simple feature selection and preprocessing
    df = df.select_dtypes(include=['number']) # Use only numeric columns for simplicity
    df = df.dropna()

    if 'Parts_Per_Hour' not in df.columns:
        st.error("Target column 'Parts_Per_Hour' not found in the dataset.")
        return None, None

    X = df.drop('Parts_Per_Hour', axis=1)
    y = df['Parts_Per_Hour']

    # Ensure we have data to train on
    if X.empty or y.empty:
        st.error("No data available for training after preprocessing. Please check the dataset.")
        return None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    # Save the trained model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    return model, r2

# --- Streamlit App ---
st.set_page_config(page_title="Manufacturing Output Prediction", layout="wide")

st.title("⚙️ Manufacturing Equipment Output Prediction")
st.write("""
This application predicts the hourly output of manufacturing equipment using a Linear Regression model.
Provide the machine parameters on the left to get a prediction.
""")

# --- Sidebar for User Input ---
st.sidebar.header("Machine Parameters")

def get_user_input():
    """
    Gets user input from the sidebar for prediction.
    """
    # Using example features from a typical manufacturing dataset.
    # IMPORTANT: These features must match the columns used for training the model.
    # For this example, we'll use a subset of numeric features.
    injection_temp = st.sidebar.slider('Injection Temperature (°C)', 180.0, 250.0, 220.0)
    injection_pressure = st.sidebar.slider('Injection Pressure (bar)', 80.0, 150.0, 115.0)
    cycle_time = st.sidebar.slider('Cycle Time (seconds)', 15.0, 45.0, 30.0)
    cooling_time = st.sidebar.slider('Cooling Time (seconds)', 8.0, 20.0, 14.0)
    material_viscosity = st.sidebar.slider('Material Viscosity (Pa·s)', 100.0, 400.0, 250.0)
    ambient_temp = st.sidebar.slider('Ambient Temperature (°C)', 18.0, 28.0, 24.0)
    machine_age = st.sidebar.slider('Machine Age (years)', 0.0, 10.0, 5.0)
    operator_experience = st.sidebar.slider('Operator Experience (years)', 0.0, 20.0, 10.0)
    maintenance_hours = st.sidebar.slider('Maintenance Hours', 0, 100, 50)
    temp_pressure_ratio = injection_temp / injection_pressure if injection_pressure != 0 else 0
    total_cycle_time = cycle_time + cooling_time
    efficiency_score = st.sidebar.slider('Efficiency Score', 0.0, 1.0, 0.5)
    machine_utilization = st.sidebar.slider('Machine Utilization', 0.0, 1.0, 0.7)


    features = {
        'Injection_Temperature': injection_temp,
        'Injection_Pressure': injection_pressure,
        'Cycle_Time': cycle_time,
        'Cooling_Time': cooling_time,
        'Material_Viscosity': material_viscosity,
        'Ambient_Temperature': ambient_temp,
        'Machine_Age': machine_age,
        'Operator_Experience': operator_experience,
        'Maintenance_Hours': maintenance_hours,
        'Temperature_Pressure_Ratio': temp_pressure_ratio,
        'Total_Cycle_Time': total_cycle_time,
        'Efficiency_Score': efficiency_score,
        'Machine_Utilization': machine_utilization,
    }
    return pd.DataFrame([features])


# --- Main Application Logic ---

# Check if model exists, if not, train it
if not os.path.exists(MODEL_PATH):
    st.info("Model not found. Training a new model...")
    with st.spinner('Training in progress...'):
        model, r2 = train_model()
    if model:
        st.success(f"Model trained successfully with an R² score of {r2:.4f}")
    else:
        st.stop()
else:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

# Get user input
input_df = get_user_input()

# Display user input
st.subheader("Input Parameters")
st.write(input_df)

# Prediction
if st.sidebar.button("Predict Output"):
    try:
        # Ensure column order is the same as during training
        # We get the columns from the model's internal state if available
        if hasattr(model, 'feature_names_in_'):
            training_cols = model.feature_names_in_
            input_df = input_df[training_cols]
        else: # Fallback for models without this attribute
            st.warning("Could not verify feature order. Prediction might be inaccurate if the order is wrong.")


        prediction = model.predict(input_df)

        st.subheader("Prediction Result")
        st.metric(label="Predicted Parts Per Hour", value=f"{prediction[0]:.2f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# --- Optional: Display Model Information ---
st.sidebar.markdown("---")
st.sidebar.info("This app uses a Linear Regression model to predict manufacturing output.")
if st.sidebar.checkbox("Show Model Details"):
    st.subheader("Model Coefficients")
    if hasattr(model, 'feature_names_in_'):
        coef_df = pd.DataFrame(model.coef_, index=model.feature_names_in_, columns=['Coefficient'])
        st.dataframe(coef_df)
    else:
        st.write("Feature names not available in the loaded model.")

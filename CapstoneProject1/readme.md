Manufacturing Equipment Output Prediction
This project contains a Streamlit web application that predicts manufacturing equipment output based on various machine parameters. The prediction is made using a Linear Regression model trained on a sample manufacturing dataset.

The application is containerized using Docker for easy deployment and scalability.

Project Structure
.
├── app.py                      # The main Streamlit application script
├── manufacturing_dataset_1000_samples.csv # Sample dataset for training
├── linear_regression_model.pkl # Saved model file (will be created on first run)
├── Dockerfile                  # Docker configuration for the application
├── requirements.txt            # Python dependencies
└── README.md                   # This README file

Getting Started
Prerequisites
Python 3.8+

Docker (for containerized deployment)

1. Running the Application Locally
A. Clone the repository or download the files.

B. Create a virtual environment and install dependencies:

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt

C. Run the Streamlit application:

Ensure that app.py and manufacturing_dataset_1000_samples.csv are in the same directory.

streamlit run app.py

The application will start, and you can access it in your web browser at http://localhost:8501.

2. Building and Running with Docker
A. Build the Docker image:

Open your terminal in the project's root directory (where the Dockerfile is located) and run the following command:

docker build -t manufacturing-prediction-app .

B. Run the Docker container:

Once the image is built, you can run it as a container:

docker run -p 8501:8501 manufacturing-prediction-app

This command maps port 8501 from the container to port 8501 on your local machine.

C. Access the application:

Open your web browser and navigate to http://localhost:8501 to use the application running inside the Docker container.

How It Works
Model Training: If a pre-trained model (linear_regression_model.pkl) is not found, the app.py script will first train a simple Linear Regression model using the data from manufacturing_dataset_1000_samples.csv. The trained model is then saved as a pickle file.

User Interface: The Streamlit interface provides sliders and input fields in the sidebar for you to enter the machine's operating parameters.

Prediction: When you click the "Predict Output" button, the application takes the input parameters, feeds them into the loaded model, and displays the predicted hourly output.
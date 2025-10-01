# Heart Disease Prediction ML Pipeline

A complete machine learning pipeline for predicting heart disease using the UCI dataset, XGBoost, and Streamlit.  
This project covers data fetching, cleaning, model training (with hyperparameter tuning), evaluation, and interactive web app deployment.

---

## 🚀 Quickstart

1. **Install all requirements:**
    ```bash
    pip install -r requirements.txt
    ```

2. **Fetch and preprocess the data:**
    ```bash
    python fetch_clean_heart_data.py
    ```

3. **Train and evaluate the XGBoost model (also saves metrics):**
    ```bash
    python train_xgboost_and_metrics.py
    ```

4. **Run the Streamlit app:**
    ```bash
    streamlit run app_streamlit.py
    ```

---

## 📁 File Structure

```
heart-disease-ml/
│
├── data/
│   └── heart_clean.csv             # Created after data fetching/preprocessing
├── models/
│   ├── best_model.pkl              # Saved trained XGBoost model
│   └── model_metrics.csv           # Model metrics (accuracy, recall, F1, confusion matrix etc.)
├── fetch_clean_heart_data.py       # Script to fetch and preprocess UCI data
├── train_xgboost_and_metrics.py    # Script to train, tune, evaluate, and save model + metrics table
├── app_streamlit.py                # Streamlit web app for prediction and metrics dashboard
├── requirements.txt                # All Python dependencies
└── README.md                       # This documentation
```

---

## 📝 Project Details

- **Data Source:** [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease) (fetched automatically with `ucimlrepo`)
- **Model:** XGBoost (with hyperparameter tuning via RandomizedSearchCV)
- **Evaluation:** Accuracy, Precision, Recall, F1 Score, ROC-AUC, Confusion Matrix (True Healthy, False Sick, etc.)
- **Web App:** Streamlit UI for interactive prediction and live metrics table (matching the format in your reference image)

---

## 📊 Output

- After prediction, the app displays the actual metrics for your model in a table, as well as a risk score for each prediction.

---

## 🔗 Reference

- UCI Heart Disease: https://archive.ics.uci.edu/ml/datasets/heart+Disease

---

## 💡 Notes

- The model and metrics will be updated every time you re-run the training script.
- If you want to try different ML models or further tune hyperparameters, edit `train_xgboost_and_metrics.py`.

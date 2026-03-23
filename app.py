from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

MODEL_PATH = "artifacts/models/model.pkl"
SCALER_PATH = "artifacts/processed/scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

FEATURES = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
]

LABELS = {
    0: "No Stroke Risk Detected",
    1: "Stroke Risk Detected",
}


ENCODERS = {
    "gender": {
        "Female": 0,
        "Male": 1,
        "Other": 2
    },
    "ever_married": {
        "No": 0,
        "Yes": 1
    },
    "work_type": {
        "Govt_job": 0,
        "Never_worked": 1,
        "Private": 2,
        "Self-employed": 3,
        "children": 4
    },
    "Residence_type": {
        "Rural": 0,
        "Urban": 1
    },
    "smoking_status": {
        "Unknown": 0,
        "formerly smoked": 1,
        "never smoked": 2,
        "smokes": 3
    }
}

FORM_OPTIONS = {
    "gender": ["Female", "Male", "Other"],
    "ever_married": ["No", "Yes"],
    "work_type": ["Govt_job", "Never_worked", "Private", "Self-employed", "children"],
    "Residence_type": ["Rural", "Urban"],
    "smoking_status": ["Unknown", "formerly smoked", "never smoked", "smokes"]
}


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    submitted_data = {}

    if request.method == "POST":
        try:
            submitted_data = request.form.to_dict()

            gender = ENCODERS["gender"][request.form["gender"]]
            age = float(request.form["age"])
            hypertension = int(request.form["hypertension"])
            heart_disease = int(request.form["heart_disease"])
            ever_married = ENCODERS["ever_married"][request.form["ever_married"]]
            work_type = ENCODERS["work_type"][request.form["work_type"]]
            residence_type = ENCODERS["Residence_type"][request.form["Residence_type"]]
            avg_glucose_level = float(request.form["avg_glucose_level"])
            bmi = float(request.form["bmi"])
            smoking_status = ENCODERS["smoking_status"][request.form["smoking_status"]]

            input_data = np.array([[
                gender,
                age,
                hypertension,
                heart_disease,
                ever_married,
                work_type,
                residence_type,
                avg_glucose_level,
                bmi,
                smoking_status
            ]])

            scaled_data = scaler.transform(input_data)
            pred = model.predict(scaled_data)[0]
            prediction = LABELS.get(pred, "Unknown")

        except Exception as e:
            error = f"Error: {str(e)}"

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        form_options=FORM_OPTIONS,
        submitted_data=submitted_data
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
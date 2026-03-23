🧠 Stroke Prediction System

A Machine Learning web application that predicts the risk of stroke based on patient health data.
Built using Python, Flask, and Scikit-learn, with a clean UI for real-time predictions.

📌 Project Overview

This project is an end-to-end Machine Learning pipeline that includes:

Data preprocessing
Handling missing values
Feature encoding
Handling imbalanced data using SMOTE
Model training and evaluation
Model serialization using Joblib
Web application using Flask

Users can input patient details through a web interface and get instant stroke risk prediction.

🛠️ Tech Stack
Python
Flask
Scikit-learn
Pandas & NumPy
imbalanced-learn (SMOTE)
HTML / CSS
Gunicorn (for deployment)
📂 Project Structure

.
├── app.py
├── requirements.txt
├── src/
│ ├── logger.py
│ └── custom_exception.py

├── templates/
│ └── index.html

├── static/
│ └── style.css

├── artifacts/
│ ├── raw/
│ │ └── data.csv
│ ├── processed/
│ │ ├── scaler.pkl
│ │ └── encoders.pkl
│ └── models/
│ └── model.pkl

└── README.md

⚙️ Installation & Setup
Clone the repository
git clone https://github.com/your-username/stroke-prediction.git

cd stroke-prediction
Create virtual environment
python -m venv venv
venv\Scripts\activate
Install dependencies
pip install -r requirements.txt
▶️ Run the Application

python app.py

Open your browser and go to:
http://127.0.0.1:5000

🌐 Deployment

This project is deployment-ready and can be deployed on:

Render
PythonAnywhere
Docker-based environments

Start command for deployment:
gunicorn app:app

📊 Features
Predict stroke risk instantly
Handles imbalanced dataset using SMOTE
Feature scaling using StandardScaler
Encodes categorical variables
Clean and responsive UI
Modular code structure
🧪 Input Features
Gender
Age
Hypertension
Heart Disease
Ever Married
Work Type
Residence Type
Average Glucose Level
BMI
Smoking Status
🎯 Output
Yes → Stroke risk detected
No → No stroke risk
⚠️ Disclaimer

This project is for educational and demonstration purposes only
and should not be used for real medical decisions.

👨‍💻 Author

Rohanta Bhamare
AI / ML Engineer

Frankfurt, Germany

LinkedIn: https://www.linkedin.com/in/rohanta-bhamare

GitHub: https://github.com/rohantabhamar

⭐ Future Improvements
Convert to Scikit-learn Pipeline
Add model explainability (SHAP)
Deploy using Docker
Add REST API (FastAPI)
Add monitoring and logging
📄 License

This project is for educational and demonstration purposes.
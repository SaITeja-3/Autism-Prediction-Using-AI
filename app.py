from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import joblib
import numpy as np

app = Flask(__name__)

# Configure SQLite Database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///autism_records.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Load the trained model
model = joblib.load("autism_model.pkl")

# Define Database Model
class AutismRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    age = db.Column(db.Float, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    jaundice = db.Column(db.String(3), nullable=False)
    autism_family = db.Column(db.String(3), nullable=False)
    q1 = db.Column(db.Integer, nullable=False)
    q2 = db.Column(db.Integer, nullable=False)
    q3 = db.Column(db.Integer, nullable=False)
    q4 = db.Column(db.Integer, nullable=False)
    q5 = db.Column(db.Integer, nullable=False)
    q6 = db.Column(db.Integer, nullable=False)
    q7 = db.Column(db.Integer, nullable=False)
    q8 = db.Column(db.Integer, nullable=False)
    q9 = db.Column(db.Integer, nullable=False)
    q10 = db.Column(db.Integer, nullable=False)
    prediction = db.Column(db.String(20), nullable=False)

# Create database
with app.app_context():
    db.create_all()

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/form')
def form():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input values
        user_input = [int(request.form[f'Q{i + 1}']) for i in range(10)]
        age = float(request.form['age'])
        gender = request.form['gender']
        jaundice = request.form['jaundice']
        autism_family = request.form['autism']

        # Convert categorical values to numeric
        gender_val = 1 if gender == 'Male' else 0
        jaundice_val = 1 if jaundice == 'Yes' else 0
        autism_family_val = 1 if autism_family == 'Yes' else 0

        # Prepare input for model
        input_data = np.array([user_input + [age, gender_val, jaundice_val, autism_family_val]])

        # Get prediction
        prediction = model.predict(input_data)[0]
        result = "Autism Detected" if prediction == 1 else "No Autism Detected"

        # Save record to database
        new_entry = AutismRecord(
            age=age,
            gender=gender,
            jaundice=jaundice,
            autism_family=autism_family,
            q1=user_input[0], q2=user_input[1], q3=user_input[2], q4=user_input[3], q5=user_input[4],
            q6=user_input[5], q7=user_input[6], q8=user_input[7], q9=user_input[8], q10=user_input[9],
            prediction=result
        )
        db.session.add(new_entry)
        db.session.commit()
        print("Data saved successfully!")
        return render_template('result.html', result=result)
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/records')
def view_records():
    entries = AutismRecord.query.all()  # Fetch all records
    return render_template('records.html', entries=entries)

if __name__ == '__main__':
    app.run(debug=True)

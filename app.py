from xml.parsers.expat import model
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        pclass = int(request.form['pclass'])
        sex = int(request.form['sex'])  # 0 for male, 1 for female
        age = float(request.form['age'])
        fare = float(request.form['fare'])
        # 0 for not embarked Q, 1 for embarked Q
        embarked_Q = int(request.form.get('embarked_Q', 0))
        # 0 for not embarked S, 1 for embarked S
        embarked_S = int(request.form.get('embarked_S', 0))
        family_size = int(request.form['family_size'])

        # Features must be in the same order as the model was trained
        features = np.array(
            [[pclass, sex, age, fare, embarked_Q, embarked_S, family_size]])
        prediction = model.predict(features)[0]

        # Map prediction to survival status
        survival_status = 'Survived' if prediction == 1 else 'Did not survive'
        return render_template('index.html', prediction_text=f'The passenger is predicted to: {survival_status}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')


if __name__ == "__main__":
    # Load the pre-trained model
    model = joblib.load('titanic_logistic_model.pkl')
    app.run(debug=True)

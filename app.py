from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import joblib
import numpy as np
import pandas as pd
import logging
from flask_session import Session

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

logging.basicConfig(level=logging.DEBUG)

# Load Models
model1 = joblib.load('./models/DecisionTreeModel.joblib')
model2 = joblib.load('./models/LinearRegressionModel.joblib')
model3 = joblib.load('./models/RandomForestModel.joblib')
model4 = joblib.load('./models/NaiveBayesModel.joblib')

# Mapping for Categorical Features
gender_mapping = {'Female': 0, 'Male': 1}
bmi_category_mapping = {'Normal': 0, 'Obese': 2, 'Overweight': 3}
sleep_disorder_mapping = {'No Disorder': 0, 'Insomnia': 1, 'Sleep Apnea': 2}

# feature_columns = [
#     'Gender', 'Age', 'Sleep Duration', 'Quality of Sleep',
#     'Physical Activity Level', 'BMI Category', 'Heart Rate', 'Daily Steps', 'Sleep Disorder'
# ]

feature_columns = [
    'Gender', 'Age', 'Sleep Duration', 'Quality of Sleep', 'Heart Rate', 'Daily Steps'
]

# Home Page
@app.route('/')
def introduction():
    return render_template('introduction.html')

# Collect User Info For Prediction
@app.route('/predict')
def home():
    return render_template('index.html')

# Predict Stress Level Based on Selected Model
@app.route('/predict/<model>', methods=['POST'])
def predict(model):
    data = request.form
    try:
        gender = gender_mapping.get(data['Gender'], -1)
        features = [
            gender,
            int(data['Age']),
            float(data['Sleep Duration']),
            int(data['Quality of Sleep']),
            int(data['Heart Rate']),
            int(data['Daily Steps'])
        ]

        feature_df = pd.DataFrame([features], columns=feature_columns)
        app.logger.debug(f"Feature DataFrame: {feature_df}")
        print("DataFrame created")

        if model == 'model1':
            prediction = model1.predict(feature_df)
        elif model == 'model2':
            prediction = model2.predict(feature_df)
        elif model == 'model3':
            prediction = model3.predict(feature_df)
        elif model == 'model4':
            prediction = model4.predict(feature_df)
        else:
            return jsonify({'error': 'Invalid model'}), 400

        prediction_value = prediction[0]
        session['prediction'] = prediction_value
        print(f"Prediction made and stored in session {prediction_value}")
        return redirect(url_for('results'))
    except KeyError as e:
        app.logger.error(f"Missing key in form data: {e}")
        return jsonify({'error': f'Missing key in form data: {e}'}), 400
    except ValueError as e:
        app.logger.error(f"Value error: {e}")
        return jsonify({'error': f'Value error: {e}'}), 400
    except Exception as e:
        app.logger.error(f"Error during prediction: {e}")
        return jsonify({'error': 'Prediction failed'}), 500

# Display Results
@app.route('/results')
def results():
    prediction = session.get('prediction')

    stress_level_explanations = [
        { 'range': [-100, 4], 'text': 'Your stress level is low, which is good for your health. Keep doing what you are doing. :)' },
        { 'range': [4, 7], 'text': 'Your stress level is moderate. It is important to manage stress to maintain good health.' },
        { 'range': [7, 100], 'text': 'Your stress level is high. Consider taking steps to reduce stress for better health. Some things you can try involve: going out with friends, reading a book, working out, and so on.' }
    ]

    explanation = next((expl['text'] for expl in stress_level_explanations if prediction >= expl['range'][0] and prediction <= expl['range'][1]), 'Unknown stress level.')

    return render_template('results.html', prediction=prediction, explanation=explanation)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)

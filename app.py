import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

model = pickle.load(open("diabetic.pkl", "rb"))

def generate_advice(features):
    advice = []

    pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age = features

    if skin_thickness > 30:
        advice.append("Your skin thickness is higher than normal. Maintaining a healthy weight could reduce the risk of diabetes.")
    if glucose > 140:
        advice.append("Your glucose level is elevated. Monitoring your diet and regular check-ups are recommended.")
    if blood_pressure > 80:
        advice.append("Your blood pressure is higher than the recommended level. Consider reducing salt intake and managing stress.")
    if insulin > 100:
        advice.append("Your insulin level is higher than average. A balanced diet and regular exercise might be beneficial.")
    if bmi > 30:
        advice.append("Your BMI indicates obesity. Weight management through a healthy diet and exercise can help reduce the risk of diabetes.")
    if pedigree > 0.5:
        advice.append("A high diabetes pedigree function suggests a strong family history. Regular screenings and a healthy lifestyle are advised.")
    if age > 45:
        advice.append("Being over 45 increases your risk of diabetes. Regular health check-ups and a balanced diet are important.")

    if not advice:
        advice.append("Your results are within the normal range, but maintaining a healthy lifestyle is always beneficial.")

    return advice

def generate_food_advice(features):
    food_advice = []

    pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age = features

    if glucose > 140:
        food_advice.append("Consider a diet low in refined sugars and carbohydrates, high in fiber, and with a focus on complex carbohydrates and lean proteins.")
    if bmi > 30:
        food_advice.append("Opt for a balanced diet rich in fruits, vegetables, lean proteins, and whole grains. Avoid high-calorie, low-nutrient foods.")
    if insulin > 100:
        food_advice.append("Include more fiber-rich foods, healthy fats, and low-GI foods.")
    if blood_pressure > 80:
        food_advice.append("Reduce sodium intake and eat more potassium-rich, magnesium-rich, and calcium-rich foods.")

    if not food_advice:
        food_advice.append("Maintain a healthy diet with a variety of foods to ensure balanced nutrition.")

    return food_advice

def generate_exercise_advice(features):
    exercise_advice = []

    pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age = features

    if glucose > 140:
        exercise_advice.append("Incorporate regular physical activity to help manage blood sugar levels. Aim for at least 30 minutes of moderate exercise most days.")
    if bmi > 30:
        exercise_advice.append("Regular exercise is crucial for managing obesity. Consider engaging in activities such as brisk walking, cycling, or swimming.")
    if insulin > 100:
        exercise_advice.append("Exercise can help improve insulin sensitivity. Try including both aerobic exercises and strength training in your routine.")
    if blood_pressure > 80:
        exercise_advice.append("Regular physical activity can help lower blood pressure. Activities like walking, jogging, and swimming are recommended.")

    if not exercise_advice:
        exercise_advice.append("Maintaining a regular exercise routine is beneficial for overall health.")

    return exercise_advice

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    data = request.form.to_dict()
    data_values = [float(data[key]) for key in ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 'dpf', 'age']]

    input_features = np.array([data_values])

    prediction = model.predict(input_features)[0]

    advice = generate_advice(data_values)
    food_advice = generate_food_advice(data_values)
    exercise_advice = generate_exercise_advice(data_values)

    result = "The model predicts that you have diabetes." if prediction == 1 else "The model predicts that you do not have diabetes."

    return render_template('index.html', prediction_text=result, tips=advice, food_tips=food_advice, exercise_tips=exercise_advice)

if __name__ == "__main__":
    app.run(debug=True)

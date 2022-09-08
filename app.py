from flask import Flask, render_template, request
import joblib
import numpy as np

model = joblib.load('ml_model_diabetes')
heartDiseaseModel = joblib.load('ml_model_heart_disease')

app = Flask(__name__)

# A must at all times
app.config['TEMPLATES_AUTO_RELOAD'] = True

app.debug = True

@app.route("/")
def home():
	return render_template('index.html')

@app.route("/diabetesdetection")
def diabetesDetection():	
	return render_template('diabetes.html')

@app.route("/diabetesdetection/predictdiabetes", methods=['POST', 'GET'])
def showDiabetesResult():
	numOfPreg = request.form['a']
	glucose = request.form['b']
	bloodPressure = request.form['c']
	skinThickness = request.form['d']
	insulin = request.form['e']
	bmi = request.form['f']
	diabPedigFunc = request.form['g']
	age = request.form['h']

	arr = np.array([[numOfPreg, glucose, bloodPressure, skinThickness, insulin, bmi, diabPedigFunc, age]], dtype=float)
	pred = model.predict(arr)

	if pred[0] == 0:
		result = "STATUS: SAFE"

	else:
		result = "STATUS: HAS DIABETES"
	
	return render_template('diabetesResult.html', data=result)


@app.route("/heartdisease-detection")
def heartDiseaseDetection():
	return render_template('heart.html')

@app.route("/heartdisease-detection/predictHeartDisease", methods=['POST', 'GET'])
def showHeartDiseaseResult():
	age = request.form['age']
	sex = request.form['sex']
	chestPainType = request.form['chestPainType']
	restingBloodPressure = request.form['restingBloodPressure']
	cholesterol = request.form['cholesterol']
	fastingBloodSugar = request.form['fastingBloodSugar']
	restingElectroResult = request.form['restingElectroResult']
	maxHeartRate = request.form['maxHeartRate']
	exInducedAngina = request.form['exInducedAngina']
	oldPeakDep = request.form['oldPeakDep']
	slope = request.form['slope']
	ca = request.form['ca']
	thaliumStress = request.form['thaliumStress']

	heartArr = np.array([[age, sex, chestPainType, restingBloodPressure, cholesterol, 
		fastingBloodSugar, restingElectroResult, maxHeartRate, exInducedAngina, 
		oldPeakDep, slope,ca, thaliumStress]], dtype=float)

	heartPred = heartDiseaseModel.predict(heartArr)

	if heartPred[0] == 0:
		result = "STATUS: SAFE"

	else:
		result = "STATUS: HAS A HEART DISEASE"


	return render_template('heartResult.html', data=result)


if __name__ == "__main__":
	app.run(debug = True)
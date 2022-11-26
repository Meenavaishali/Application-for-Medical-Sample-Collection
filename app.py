from flask import Flask,request,render_template,url_for,redirect
import numpy as np 
import pickle

# Initializing the Flask API
app = Flask(__name__)

#loading the model
model = pickle.load(open('regression_model.pkl','rb'))

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

	if request.method == 'POST':
		feature_1 = request.form.get('Agent_ID',False)
		feature_2 = request.form.get('pincode',False)
		feature_3 = request.form.get('Diagnostic_Centers',False)
		feature_4 = request.form.get('Time_slot',False)
		feature_5 = request.form.get('shortest_distance_Agent_Pathlab_m',False)
		feature_6 = request.form.get('shortest_distance_Patient_Pathlab_m',False)
		feature_7 = request.form.get('shortest_distance_Patient_Agent_m',False)
		feature_8 = request.form.get('Availabilty_time_Patient',False)
		feature_9 = request.form.get('Age',False)
		feature_10 = request.form.get('Gender',False)
		feature_11 = request.form.get('Test_name',False)
		feature_12 = request.form.get('Sample',False)
		feature_13 = request.form.get('Way_Of_Storage_Of_Sample',False)
		feature_14 = request.form.get('Time_For_Sample_Collection_MM',False)
        
		feature_15 = request.form.get('Time_Agent_Pathlab_sec',False)


		#converting all the features into float if they were to receive the integer values 
		final_features = np.array([feature_1,feature_2,feature_3,feature_4,feature_5,feature_6,feature_7,feature_8,feature_9,feature_10,feature_11,feature_12,feature_13,feature_14,feature_15],np.float64).reshape(1,-1) #converting the form values into numpy array
		prediction = model.predict(final_features)/60
		output = round(prediction[0], 1)
		
		
		
		return render_template('index.html',pincode=feature_1,Diagnostic_Centers=feature_2,Gender=feature_3,Sample = feature_4,Test_name=feature_5,Way_Of_Storage_Of_Sample=feature_6,Time_slot=feature_7,Availabilty_time_Patient=feature_8,shortest_distance_Agent_Pathlab_m=feature_9,shortest_distance_Patient_Pathlab_m=feature_10,shortest_distance_Patient_Agent_m=feature_11,Time_For_Sample_Collection_MM=feature_12,Time_Agent_Pathlab_sec=feature_13,Age=feature_14, prediction_text ="Agent will reach within {} minutes".format(output))
	else:
		return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)
    

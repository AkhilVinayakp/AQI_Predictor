from flask import Flask,render_template,request
import pickle
#import numpy as np
import pandas as pd
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict',methods=['POST'])
def predict():
   City= int(request.values['Citydrop'])
   PM25= float(request.values['PM2.5'])
   PM10= float(request.values['PM10'])
   NH3= float(request.values['NH3'])
   CO= float(request.values['CO'])
   SO2= float(request.values['SO2'])
   O3= float(request.values['O3'])
   Nitrites= float(request.values['Nitrites'])
   month= int(request.values['monthdrop'])
   Day= int(request.values['Day'])
   
   features=[City,PM25,PM10,NH3,CO,SO2,O3,Nitrites,month,Day]
   test2= pd.DataFrame([features],columns= 
   ['City', 'PM2.5', 'PM10', 'NH3', 'CO', 'SO2', 'O3', 'Nitrites', 'month', 'Day'],dtype=float)
   #p=model.predict(test2)
   #arr=[np.array(test2)]
   output=model.predict(test2)
   output=output.item()
   output=round(output)
   #print(output)
   def get_AQI_bucket(x):
    if x <= 50:
        return "Good"
    elif x <= 100:
        return "Satisfactory"
    elif x <= 200:
        return "Moderate"
    elif x <= 300:
        return "Poor"
    elif x <= 400:
        return "Very Poor"
    elif x > 400:
        return "Severe"
    else:
        return 0
    
   AQI_status=get_AQI_bucket(output)
   return render_template ('result.html',prediction_text="The AQI index is :{}".format(output),prediction_status="The AQI Status is :{}".format(AQI_status))
   
if __name__=='__main__':
    app.run(port=5000)


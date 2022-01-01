# AQI prediction model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA
import pickle
from sklearn.model_selection import train_test_split

#reading data set
data=pd.read_excel('city_day_final (1).xlsx',parse_dates=['Date'])

#Filling missing values

pmean=data["PM2.5"].mean()
data["PM2.5"].fillna(pmean,inplace=True)
pmmean=data["PM10"].mean()
data["PM10"].fillna(pmmean,inplace=True)
nomean=data["NO"].mean()
data["NO"].fillna(nomean,inplace=True)
nomean=data["NO2"].mean()
data["NO2"].fillna(nomean,inplace=True)
noxmean=data["NOx"].mean()
data["NOx"].fillna(noxmean,inplace=True)
nh3mean=data["NH3"].mean()
data["NH3"].fillna(nh3mean,inplace=True)
comean=data["CO"].mean()
data["CO"].fillna(comean,inplace=True)
so2mean=data["SO2"].mean()
data["SO2"].fillna(so2mean,inplace=True)
o3mean=data["O3"].mean()
data["O3"].fillna(pmean,inplace=True)
bmean=data["Benzene"].mean()
data["Benzene"].fillna(bmean,inplace=True)
tmean=data["Toluene"].mean()
data["Toluene"].fillna(tmean,inplace=True)
xmean=data["Xylene"].mean()
data["Xylene"].fillna(xmean,inplace=True)

# Filling missing values in AQI column using calculating subindexes

def get_PM25_subindex(x):
    if x <= 30:
        return x * 50 / 30
    elif x <= 60:
        return 50 + (x - 30) * 50 / 30
    elif x <= 90:
        return 100 + (x - 60) * 100 / 30
    elif x <= 120:
        return 200 + (x - 90) * 100 / 30
    elif x <= 250:
        return 300 + (x - 120) * 100 / 130
    elif x > 250:
        return 400 + (x - 250) * 100 / 130
    else:
        return 0
data["PM2.5_SubIndex"] = data["PM2.5"].astype(int).apply(lambda x: get_PM25_subindex(x))
    
def get_PM10_subindex(x):
    if x <= 50:
        return x
    elif x <= 100:
        return x
    elif x <= 250:
        return 100 + (x - 100) * 100 / 150
    elif x <= 350:
        return 200 + (x - 250)
    elif x <= 430:
        return 300 + (x - 350) * 100 / 80
    elif x > 430:
        return 400 + (x - 430) * 100 / 80
    else:
        return 0
data["PM10_SubIndex"] = data["PM10"].astype(int).apply(lambda x: get_PM10_subindex(x))

def get_SO2_subindex(x):
    if x <= 40:
        return x * 50 / 40
    elif x <= 80:
        return 50 + (x - 40) * 50 / 40
    elif x <= 380:
        return 100 + (x - 80) * 100 / 300
    elif x <= 800:
        return 200 + (x - 380) * 100 / 420
    elif x <= 1600:
        return 300 + (x - 800) * 100 / 800
    elif x > 1600:
        return 400 + (x - 1600) * 100 / 800
    else:
        return 0
data["SO2_SubIndex"] = data["SO2"].astype(int).apply(lambda x: get_SO2_subindex(x))

def get_NOx_subindex(x):
    if x <= 40:
        return x * 50 / 40
    elif x <= 80:
        return 50 + (x - 40) * 50 / 40
    elif x <= 180:
        return 100 + (x - 80) * 100 / 100
    elif x <= 280:
        return 200 + (x - 180) * 100 / 100
    elif x <= 400:
        return 300 + (x - 280) * 100 / 120
    elif x > 400:
        return 400 + (x - 400) * 100 / 120
    else:
        return 0
data["NOx_SubIndex"] = data["NOx"].astype(int).apply(lambda x: get_NOx_subindex(x))  

def get_NH3_subindex(x):
    if x <= 200:
        return x * 50 / 200
    elif x <= 400:
        return 50 + (x - 200) * 50 / 200
    elif x <= 800:
        return 100 + (x - 400) * 100 / 400
    elif x <= 1200:
        return 200 + (x - 800) * 100 / 400
    elif x <= 1800:
        return 300 + (x - 1200) * 100 / 600
    elif x > 1800:
        return 400 + (x - 1800) * 100 / 600
    else:
        return 0
data["NH3_SubIndex"] = data["NH3"].astype(int).apply(lambda x: get_NH3_subindex(x)) 

def get_CO_subindex(x):
    if x <= 1:
        return x * 50 / 1
    elif x <= 2:
        return 50 + (x - 1) * 50 / 1
    elif x <= 10:
        return 100 + (x - 2) * 100 / 8
    elif x <= 17:
        return 200 + (x - 10) * 100 / 7
    elif x <= 34:
        return 300 + (x - 17) * 100 / 17
    elif x > 34:
        return 400 + (x - 34) * 100 / 17
    else:
        return 0
data["CO_SubIndex"] = data["CO"].astype(int).apply(lambda x: get_CO_subindex(x))

def get_O3_subindex(x):
    if x <= 50:
        return x * 50 / 50
    elif x <= 100:
        return 50 + (x - 50) * 50 / 50
    elif x <= 168:
        return 100 + (x - 100) * 100 / 68
    elif x <= 208:
        return 200 + (x - 168) * 100 / 40
    elif x <= 748:
        return 300 + (x - 208) * 100 / 539
    elif x > 748:
        return 400 + (x - 400) * 100 / 539
    else:
        return 0
data["O3_SubIndex"] = data["O3"].astype(int).apply(lambda x: get_O3_subindex(x))

data['AQI']=data['AQI'].fillna(round(data[['PM2.5_SubIndex','PM10_SubIndex','SO2_SubIndex','NOx_SubIndex','NH3_SubIndex','CO_SubIndex','O3_SubIndex']].max(axis=1)))

# Filling AQI bucket column
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
data["AQI_Bucket"] = data['AQI_Bucket'].fillna(data['AQI'].apply(lambda x: get_AQI_bucket(x)))

data_new=data.drop(['StationId','StationName'],axis=1)
data_new=data_new.drop([ 'PM2.5_SubIndex', 'PM10_SubIndex',
       'SO2_SubIndex', 'NOx_SubIndex', 'NH3_SubIndex', 'CO_SubIndex',
       'O3_SubIndex'],axis=1)
data_new=data_new.drop(['Benzene','Toluene','Xylene'],axis=1)
data_new['Nitrites']=data_new['NO']+data_new['NO2']+data_new['NOx']
data_new=data_new.drop(['NO','NO2','NOx'],axis=1)
data_new['month'] = data_new['Date'].dt.month
data_new['Day'] = data_new['Date'].dt.day
from sklearn.preprocessing import LabelEncoder
label_en=LabelEncoder()
data_new['City']=label_en.fit_transform(data_new['City'])
data_new['State']=label_en.fit_transform(data_new['State'])
data_new.set_index('Date',inplace=True)
print(data_new.head())

X=data_new.drop(['State','AQI','AQI_Bucket'],axis=1)
X=X.loc['2015-01-01':'2019-12-30']
print(X.columns)
Y=data_new[['AQI']]
Y=Y.loc['2015-01-01':'2019-12-30']

# Splitting data to test and train data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,random_state=0,test_size=0.2,shuffle=False)


import xgboost as xgb
model = xgb.XGBRegressor(max_depth=5,objective='reg:squarederror',n_estimators=200,learning_rate=0.05,random_state=1,booster='dart')
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

import math
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
print("MSE of Extreme Gradient boosting regressor is :",mean_squared_error(y_test,y_pred))
print("R squared value of Extreme Gradient boosting regressor is :",r2_score(y_test,y_pred))
print("MAE of Extreme Gradient boosting regressor is :",mean_absolute_error(y_test,y_pred))
print("RMSE of Extreme Gradient boosting regressor is",math.sqrt(mean_squared_error(y_test,y_pred)))

pickle.dump(model,open('model.pkl','wb') )





import pandas as pd
import json
data = pd.read_csv("features.csv")
data_conf = {}
cols = data.columns
data_conf['cols'] = list(cols)
des =  data.describe()
data_conf['pm2.5'] = {'min': des['PM2.5']['min'],'max': des['PM2.5']['max']}
data_conf['PM10'] = {'min': des['PM10']['min'],'max': des['PM10']['max']}
data_conf['NH3'] = {'min': des['NH3']['min'],'max': des['NH3']['max']}
data_conf['CO'] = {'min': des['CO']['min'],'max': des['CO']['max']}
data_conf['SO2'] = {'min': des['SO2']['min'],'max': des['SO2']['max']}
data_conf['O3'] = {'min': des['O3']['min'],'max': des['O3']['max']}
data_conf['Nitrites'] = {'min': des['Nitrites']['min'],'max': des['Nitrites']['max']}
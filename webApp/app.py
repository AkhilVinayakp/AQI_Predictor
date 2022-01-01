from attr import s
from numpy import min_scalar_type, mod
import pandas as pd
import streamlit as slt
import json
import pickle
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
# setting the title 
slt.title("AQI Predictor")
# slt.write(
#     pd.DataFrame({
#         'A':[2,1,3],
#         'B':[45,6,8]
#     })
# )

class AQI:
    def __init__(self) -> None:
        self.view = "default"
        # loadding data configs
        with open('data_conf2.pkl', 'rb') as fp:
            self.config = pickle.load(fp) 
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.model = pickle.load(open('model.pkl', 'rb'))
        self.data_vis = pd.read_csv('plotting_data.csv')
        
    def construct_sidebar(self):
        cols = [col for col in self.config.get('cols')]

        slt.sidebar.markdown(
            '<p class="header-style">Set the Features</p>',
            unsafe_allow_html=True
        )
        cities = slt.sidebar.selectbox(
            f"Select City",
            self.config.get('cities'))
        month = slt.sidebar.selectbox("Select Month", self.config.get('month'))
        day = slt.sidebar.selectbox("Select Day", self.config.get('day'))
        pm2 = slt.sidebar.slider("PM2.5", min_value=int(self.config.get('pm2.5', {}).get('min')), max_value=int(self.config.get('pm2.5',{}).get('max')))
        pm10 = slt.sidebar.slider("PM10", min_value=int(self.config.get('PM10',{}).get('min')), max_value=int(self.config.get('PM10',{}).get('max')))
        nh3 = slt.sidebar.slider("NH3", min_value=int(self.config.get('NH3',{}).get('min')), max_value=int(self.config.get('NH3',{}).get('max')))
        co = slt.sidebar.slider("CO", min_value=int(self.config.get('CO',{}).get('min')), max_value=int(self.config.get('CO',{}).get('max')))
        so2 = slt.sidebar.slider("SO2", min_value=int(self.config.get('SO2',{}).get('min')), max_value=int(self.config.get('SO2',{}).get('max')))
        o3 = slt.sidebar.slider("O3", min_value=int(self.config.get('O3',{}).get('min')), max_value=int(self.config.get('O3',{}).get('max')))
        nit = slt.sidebar.slider('Nitrites', min_value=int(self.config.get('Nitrites',{}).get('min')), max_value=int(self.config.get('Nitrites',{}).get('max')))

        values = {'cities': cities, 'month': month, 'day': day, 'pm2': pm2, 'pm10': pm10, 'nh3':nh3, 'co': co, 'so2': so2, 'o3': o3, 'nit': nit}
        values = pd.DataFrame([values])
        # values = self.scaler.fit_transform(values)
        return values
    @staticmethod
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
    def viz_one(self):
        ''''''
        data_final = self.data_vis
        df_aqi_trend_year=data_final.groupby(['City','year','AQI'])['AQI'].mean()
        fig = plt.figure(figsize=(10, 4))
        sns.lineplot(data=df_aqi_trend_year,x='year',y='AQI',hue='City')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        return plt
    def vix_two(self, select_city):
        '''visualization based on cities'''
        f_data = self.data_vis[['City', 'year', 'AQI']]
        f_data = f_data[f_data['City'] == select_city]
        f_data = f_data[['year','AQI']].groupby('year').mean()
        return f_data

    def load_default_view(self):
        slt.subheader("Adjust the settings to view the prediction")
        values = self.construct_sidebar()
        outcome = self.model.predict(values)
        outcome = outcome.item()
        outcome = round(outcome)

        aqi_stats = AQI.get_AQI_bucket(outcome)
        slt.markdown(
            f'<h5><center> Predicted AQI: {outcome}</center> \
               <h6> <center> Status {aqi_stats} </center>',
               unsafe_allow_html= True
        )
        selected = slt.radio(label = 'select an Option', options = ['R1','City','R3'])
        slt.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        # slt.write(f'selected{selected}')
        if selected == "R1":
            with slt.spinner("Loading"):
                plot = self.viz_one()
                slt.pyplot(plot)
        if selected == "City":
            sel_city = slt.selectbox("select a city", self.data_vis['City'].unique())
            df = self.vix_two(sel_city)
            slt.line_chart(df)



        
        
        
            



aqi = AQI()
aqi.load_default_view()



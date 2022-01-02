from attr import s
from numpy import min_scalar_type, mod
import pandas as pd
import streamlit as slt
import json
import pickle
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit.type_util import data_frame_to_bytes
from statsmodels.tsa.seasonal import seasonal_decompose
# setting the title 

PAGE_CONFIG = {"page_title" : "Predict AQI", "page_icon":":chart_with_upwards_trend", "layout": "centered"}
slt.set_page_config(**PAGE_CONFIG)
slt.title("AQI Predictor")
# slt.write(
#     pd.DataFrame({
#         'A':[2,1,3],
#         'B':[45,6,8]
#     })
# )
city_name_map = {'Ahmedabad': 0,
    'Aizawl': 1,
    'Amaravati': 2,
    'Amritsar': 3,
    'Bengaluru': 4,
    'Bhopal': 5,
    'Brajrajnagar': 6,
    'Chandigarh': 7,
    'Chennai': 8,
    'Coimbatore': 9,
    'Delhi': 10,
    'Ernakulam': 11,
    'Gurugram': 12,
    'Guwahati': 13,
    'Hyderabad': 14,
    'Jaipur': 15,
    'Jorapokhar': 16,
    'Kochi': 17,
    'Kolkata': 18,
    'Lucknow': 19,
    'Mumbai': 20,
    'Patna': 21,
    'Shillong': 22,
    'Talcher': 23,
    'Thiruvananthapuram': 24,
    'Visakhapatnam': 25}
class AQI:
    def __init__(self) -> None:
        self.view = "default"
        # loadding data configs
        with open('webApp/data_conf2.pkl', 'rb') as fp:
            self.config = pickle.load(fp) 
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.model = pickle.load(open('webApp/model.pkl', 'rb'))
        self.data_vis = pd.read_csv('webApp/plotting_data.csv', parse_dates=['Date'])
        
    def construct_sidebar(self):
        city_names = city_name_map.keys()
        slt.sidebar.markdown(
            '<p class="header-style">Set the Features</p>',
            unsafe_allow_html=True
        )
        cities = slt.sidebar.selectbox(
            f"Select City", city_names)
        cities = city_name_map.get(cities)
        month = slt.sidebar.selectbox("Select Month", self.config.get('month'))
        if month in [4, 6, 9, 11]:
            day_range = list(range(1,31)) 
        elif month == 2:
            day_range = list(range(1,29))
        else:
            day_range = list(range(1,32))
        day = slt.sidebar.selectbox("Select Day", day_range)
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
        selected = slt.radio(label = 'select an Option', options = ['Places & AQI','City wise AQI','AQI India', 'seasonality and trend', "Top 10 Polluted Cities"])
        slt.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        # slt.write(f'selected{selected}')
        if selected == 'Places & AQI':
            with slt.spinner("Loading"):
                plot = self.viz_one()
                slt.pyplot(plot)
        if selected == "City wise AQI":
            sel_city = slt.selectbox("select a city", self.data_vis['City'].unique())
            with slt.spinner("Loading graph"):
                df = self.vix_two(sel_city)
                slt.line_chart(df)
        if selected == "AQI India":
            data_final = self.data_vis
            with slt.spinner("Constructing graph...."):
                cities_all = data_final.pivot_table(values='AQI', index=['Date'], columns='City')
                cities_all=cities_all.add_suffix('_AQI')
                cities=cities_all.resample(rule='MS').mean()
                cities['India_AQI']=cities.mean(axis=1)
                cities.reset_index()
                slt.line_chart(cities["India_AQI"])
        if selected == 'seasonality and trend':
            data_final = self.data_vis
            with slt.spinner("loading graph...."):
                cities_all = data_final.pivot_table(values='AQI', index=['Date'], columns='City')
                cities_all=cities_all.add_suffix('_AQI')
                cities=cities_all.resample(rule='MS').mean()
                cities['India_AQI']=cities.mean(axis=1)
                cities.reset_index()
                fig = seasonal_decompose(cities['India_AQI'], model='additive').plot()
                slt.pyplot(fig)
        if selected == "Top 10 Polluted Cities":
            data_final = self.data_vis
            with slt.spinner("loading charts"):
                pollutants=['PM','Nitric','BTX','NH3','CO', 'SO2', 'O3']
                for i in pollutants:
                    df=data_final[['City',i]].groupby('City').mean().sort_values(i,ascending=False).iloc[:10,:]
                    slt.bar_chart(df)

        
        
        
            



aqi = AQI()
aqi.load_default_view()



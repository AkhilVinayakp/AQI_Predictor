import pandas as pd
import pickle
import os
import logging


# setting up logging config
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class InputStream:
    def __init__(self) -> None:
        try:
            self.org_data: pd.DataFrame = pd.read_csv("../data/city_day_final.csv")
        except:
            logging.error('can not find the file')
    
    def data_info(self, skip_preprocess_pipeling= False):
        if skip_preprocess_pipeling:
            pass
        else:
            logging.info(self.org_data.info())




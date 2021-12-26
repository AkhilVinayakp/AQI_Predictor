from attr import s
import pandas as pd
import streamlit as slt

# setting the title 
slt.title("Sample testing App")
# slt.write(
#     pd.DataFrame({
#         'A':[2,1,3],
#         'B':[45,6,8]
#     })
# )

class AQI:
    def __init__(self) -> None:
        self.view = "default"

    def default_view(self):
        with slt.sidebar:
            pass
            



aqi = AQI()
aqi.default_view()



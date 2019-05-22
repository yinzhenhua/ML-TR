# -*- coding:utf-8 -*-
"""
读取数据
"""

import pandas as pd

def read_aqi():
    aqi_data = pd.read_csv("aqi2.csv")
    label = aqi_data["AQI"].values
    x = aqi_data[['PM2.5', 'PM10', 'CO', 'No2', 'So2', 'O3']].values
    return x.reshape(-1, 6), label.reshape(-1, 1)

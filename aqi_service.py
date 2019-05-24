# -*- coding:utf-8 -*-
"""
将AQI服务通过Flask发出至公网，供其他程序调用
"""

from flask import Flask, request
from flask_cors import CORS
import read_data
import numpy as np
import json

APP = Flask(__name__)
CORS(APP)

@APP.route("/<name>")
def index(name):
    return "hello" + name

@APP.route("/aqi", methods=['GET','POST'])
def get_aqi_value():
    """
    根据用户提供的输入数据，完成aqi值的预测
    """
    json_data = request.get_data()
    json_data = json.loads(json_data.decode('utf-8'))
    # PM2.5,PM10,CO,No2,So2,O3
    pm25 = json_data.get('pm25')
    pm10 = json_data.get('pm10')
    co = json_data.get('co')
    no2 = json_data.get('no2')
    so2 = json_data.get('so2')
    o3 = json_data.get('o3')
    input_data = [pm25, pm10, co, no2, so2, o3]

    x = np.array(input_data)
    x = read_data.standard_data(x)
    # 从文件中读取theta
    with open('model.txt', 'r') as f:
        theta = np.array([float(line) for line in f.readlines()]).reshape(6, 1)
    aqi_value = np.dot(x, theta)
    return json.dumps({'result':aqi_value[0]})


if __name__ == "__main__":
    APP.run()

# -*- coding:utf-8 -*-
"""
是为了演示梯度下降，梯度下降有几个重要因素：
1. 梯度
2. 负梯度
3. 学习率
"""
import numpy as np
import read_data
from show import show_cost

def get_grad(theta, x, y):
    """
    x:矩阵
    y:矩阵
    theta:矩阵
    """
    grad = np.dot(np.transpose(x), (np.dot(x, theta)-y))
    return grad

def gradient_descending(theta, x, y, learning_rate):
    costs = []
    for _ in range(200):
        theta = theta - get_grad(theta,x,y)*learning_rate
        costs.append(get_cost(theta, x, y))
    show_cost(costs)

def get_cost(theta, x, y):
    """
    x:是一个矩阵
    y:是一个矩阵
    theta:是一个矩阵
    """
    return np.mean((np.dot(x, theta) - y) ** 2)*0.5


x, y = read_data.read_aqi()
theta = np.zeros((6 ,1))
learning_rate = 0.001
gradient_descending(theta, x, y ,learning_rate)

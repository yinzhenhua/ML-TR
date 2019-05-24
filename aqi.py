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

def gradient_descending(theta, x, y,v_x, v_y, learning_rate):
    """
    通过梯度下降算法，对线性回归模型进行训练
    """
    train_costs = []     # 记录训练过程中产生的cost
    validation_costs=[]    # 记录验证集上产生的cost
    for _ in range(200):
        theta = theta - get_grad(theta,x,y)*learning_rate
        train_costs.append(get_cost(theta, x, y))
        validation_costs.append(get_cost(theta, v_x, v_y))
    show_cost(train_costs, validation_costs)
    # TODO: 将theta的值保存起来
    with open('model.txt', 'w') as f:
        for i in theta:
            for j in i:
                f.write(str(j)+"\n")
    return theta


def test_model(theta, test_x, test_y):
    """
    使用R方误差来测试模型的优劣
    """
    r = 1 - get_cost(theta, test_x, test_y)/np.var(test_y)
    print(r)


def get_cost(theta, x, y):
    """
    x:是一个矩阵
    y:是一个矩阵
    theta:是一个矩阵
    """
    return np.mean((np.dot(x, theta) - y) ** 2)


'''
train_data, validation_data, test_data = read_data.read_aqi()

theta = np.zeros((6 ,1))
learning_rate = 0.00001
theta = gradient_descending(theta, train_data[0], train_data[1] , validation_data[0], validation_data[1], learning_rate)
#test_model(theta, test_data[0], test_data[1])
'''

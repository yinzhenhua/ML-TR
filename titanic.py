# -*- coding:utf-8 -*-
"""
泰坦尼号生存率预测
"""
import data_reader
import numpy as np
import show


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def get_grad(theta, x, y):
    h = sigmoid(np.dot(x, theta))
    return np.dot(np.transpose(x), h - y)


def get_cost(theta, x, y):
    h = sigmoid(np.dot(x, theta))
    return -1 * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))


def batch_gradient_descending(theta, x, y, learning_rate, batch_size, ephocs):
    costs = []
    rows = x.shape[0]
    batches = rows // batch_size
    if rows % batch_size != 0:
        batches += 1
    for ephoc in range(ephocs):
        for batch in range(batches):
            start = batch * batch_size % rows
            end = min(start + batch_size, rows)
            t_x = x[start:end]
            t_y = y[start:end]
            theta = theta - learning_rate * get_grad(theta, t_x, t_y)
            cost = get_cost(theta, x, y)
            costs.append(cost)
        # 使用学习率衰减模型，更新迭代学习率
        learning_rate = learning_rate / (1 + 0.99 * ephoc)
    show.show_cost(costs)


x, y = data_reader.read()
theta = np.random.rand(x.shape[1], 1)
learning_rate = 0.1
batch_size=100
batch_gradient_descending(theta, x, y, learning_rate, batch_size, 20)

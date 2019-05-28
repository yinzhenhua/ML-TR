# -*- coding:utf-8 -*-

import seaborn as sns
import matplotlib.pyplot as plt

def show_cost(costs):
    sns.set_style("whitegrid")
    plt.plot(costs)
    plt.show()

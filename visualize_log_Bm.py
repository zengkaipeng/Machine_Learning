import os
import json
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    Fms = []
    for i in range(1, 9):
        with open(f'result/logistic_kpart/Cross_point_{i}.json') as Fin:
            INFO = json.load(Fin)
        Fmean = sum(INFO.values()) / 10
        Fms.append(Fmean)
    plt.plot(list(range(1, 9)), Fms, label='$Bm$')
    plt.xlabel('$\\frac{n_{neg}}{n_{pos}}$')
    plt.ylabel('$Bm$')
    Xs = np.arange(1, 8, 0.01)
    Ys = np.log(1 / Xs)
    plt.plot(Xs, Ys, label='y=$-\\log x$')
    plt.legend()
    plt.show()

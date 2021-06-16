import numpy as np
import matplotlib.pyplot as plt
import json

train_labels = np.load('train_labels.npy')
extra_labels = np.load('extra_labels.npy')
train_labels = np.concatenate([train_labels, extra_labels])
Xnums = np.array([np.sum(train_labels == x) for x in range(10)])
train_len = len(train_labels)
Xnums = Xnums / - (Xnums - train_len)

fig = plt.figure()
ax = fig.add_subplot(111)

ax2 = ax.twinx()

with open('result/logistic_kpart/Cross_point_9.json') as Fin:
    INFO = json.load(Fin)

Ys = [INFO[str(x)] for x in range(10)]

lin1 = ax.plot(list(range(10)), Xnums, label='$\\frac{n_{pos}}{n_{neg}}$', color='blue')
lin2 = ax2.plot(list(range(10)), Ys, label='Boundary', color='orange')
lins = lin1 + lin2
labs = [l.get_label() for l in lins]
ax.legend(lins, labs)
ax.set_xlabel('Number')
ax2.set_ylabel('Bias of Boundary')
ax.set_ylabel('$\\frac{n_{pos}}{n_{neg}}$')
plt.tight_layout()
plt.savefig('corr.png')
import matplotlib.pyplot as plt
import os
import json

if __name__ == '__main__':
    if not os.path.exists('Visualize'):
        os.mkdir("Visualize")

    with open('result/CNN1/Results.json') as Fin:
    	INFO = json.load(Fin)

    plt.clf()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    Xs = list(range(20))
    plt.plot(Xs, INFO['trainloss'], label='Train Loss')
    plt.plot(Xs, INFO['testloss'], label='Test Loss')
    plt.legend()
    plt.savefig('Visualize/Loss.png')
    plt.clf()

    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.plot(Xs, INFO['trainacc'], label='Train Accuracy')
    plt.plot(Xs, INFO['testacc'], label='Test Accuracy')
    plt.legend()
    plt.savefig('Visualize/Accuracy.png')


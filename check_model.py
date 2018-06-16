from collections import Counter
from time import time
import pickle
from utils import MLP2
import mnist
import numpy as np
import matplotlib.pyplot as plt


def main(a, b):
    print 'load model'
    # model_path = 'ea_params/model_{}_{}.params'.format(a, b)
    model_path = 'ea_params/model_best.params'
    params = pickle.load(open(model_path))
    mlp = MLP2(params=params)

    print 'load data'
    mndata = mnist.MNIST('./data')
    test_x, test_y = mndata.load_testing()
    test_x = np.array(test_x).astype('float32') / 255

    preds = []
    good = 0.0
    total_loss = 0.0
    for x, y in zip(test_x, test_y):
        y_hat = mlp.forward(x)
        pred = np.argmax(y_hat)
        preds.append(pred)

        if y == pred:
            good += 1
        total_loss += -np.log(y_hat[y])

    print 'accuracy:', good / len(test_x), 'loss:', total_loss / len(test_x)

    print 'show preds counter'
    preds_counter = Counter(preds)
    plt.bar(range(len(preds_counter)), preds_counter.values(), align='center')
    plt.xticks(range(len(preds_counter)), preds_counter.keys())
    plt.show()


if __name__ == '__main__':
    t0 = time()
    main(90, 90)
    print 'time to run:', time() - t0

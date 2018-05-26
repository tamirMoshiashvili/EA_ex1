from time import time
import numpy as np
import mnist
import pickle

log = open('log.txt', 'w')
log.write('train-loss,dev-accuracy\n')


def softmax(x):
    """
    Compute the softmax vector.
    x: a n-dim vector (numpy array)
    returns: an n-dim vector (numpy array) of softmax values
    """
    x -= np.max(x)  # For numeric stability
    x = np.exp(x)
    x /= np.sum(x)

    return x


def sigmoid(x):
    """
    Compute the sigmoid vector.
    x: a n-dim vector (numpy array)
    returns: an n-dim vector (numpy array) of sigmoid values.
    """
    return np.array([1 / (1 + np.exp(-i)) for i in x])


class MLP1(object):
    def __init__(self, in_dim, hid_dim1, hid_dim2, out_dim):
        # Xavier Glorot init
        Glorot_init = lambda n, m: np.random.uniform(-np.sqrt(6.0 / (n + m)), np.sqrt(6.0 / (n + m)),
                                                     (n, m) if (n != 1 and m != 1) else n * m)

        self.W3 = Glorot_init(hid_dim2, out_dim)
        self.W2 = Glorot_init(hid_dim1, hid_dim2)
        self.W1 = Glorot_init(in_dim, hid_dim1)
        self.b1 = Glorot_init(1, hid_dim1)
        self.b2 = Glorot_init(1, hid_dim2)
        self.b3 = Glorot_init(1, out_dim)

    def forward(self, x):
        """
        :param x: numpy array of size in_dim.
        :return: numpy array of size out_dim.
        """
        sig1 = sigmoid(np.dot(x, self.W1) + self.b1)
        sig2 = sigmoid(np.dot(sig1, self.W2) + self.b2)
        return softmax(np.dot(sig2, self.W3) + self.b3)

    def predict_on(self, x):
        """
        :param x: numpy array of size in_dim.
        :return: scalar to indicate the predicted label of x.
        """
        return np.argmax(self.forward(x))

    def loss_and_gradients(self, x, y):  # TODO
        """
        :param x: numpy array of size in_dim.
        :param y: scalar, label of x.
        :return: loss (float) and gradients (list of size 4).
        """
        sig1 = sigmoid(np.dot(x, self.W1) + self.b1)
        sig2 = sigmoid(np.dot(sig1, self.W2) + self.b2)
        y_hat = softmax(np.dot(sig2, self.W3) + self.b3)
        loss = -np.log(y_hat[y])  # NLL loss

        # gradient of b3
        gb3 = np.copy(y_hat)
        gb3[y] -= 1

        # gradient of W3
        gW3 = np.outer(sig2, y_hat)
        gW3[:, y] -= sig2

        # gradient of b2 - use the chain rule
        dloss_dsigmoid2 = -self.W3[:, y] + np.dot(self.W3, y_hat)
        dsigmoid2_db2 = sig2 * (1 - sig2)
        gb2 = dloss_dsigmoid2 * dsigmoid2_db2

        # gradient of W2 - use the chain rule
        gW2 = np.outer(sig1, gb2)

        # gradient of b1 - use the chain rule
        dsig1_db1 = sig1 * (1 - sig1)
        gb1 = np.dot(self.W2, gb2) * dsig1_db1

        # gradient of W1
        gW1 = np.outer(x, gb1)

        return loss, [gW3, gW2, gW1, gb1, gb2, gb3]

    def accuracy_on_dataset(self, dataset):
        """
        :param dataset: list of tuples, each is (x, y) where x is vector and y is its label.
        :return: accuracy of the model on the given dataset, float between 0 to 1.
        """
        good = 0.0
        for x, y in dataset:
            y_prediction = self.predict_on(x)
            if y_prediction == y:
                good += 1
        return good / len(dataset)

    def get_params(self):
        """
        :return: list of model parameters.
        """
        return [self.W3, self.W2, self.W1, self.b1, self.b2, self.b3]

    def set_params(self, params):
        """
        :param params: list of size 4.
        """
        self.W3 = params[0]
        self.W2 = params[1]
        self.W1 = params[2]
        self.b1 = params[3]
        self.b2 = params[4]
        self.b3 = params[5]


def train_classifier(train_data, dev_data, model,
                     num_epochs=10, learning_rate=0.01, batch_size=8):
    """
    train the model on the given train-set, evaluate its performance on dev-set.
    after training, the best parameters of the model will be set to the model.
    :param train_data: array-like of tuples, each is (x, y) where x is numpy array and y is its label.
    :param dev_data: array-like of tuples, each is (x, y) where x is numpy array and y is its label.
    :param model: NN model.
    :param num_epochs: number of epochs.
    :param learning_rate: float.
    :param batch_size: size of batch.
    """
    best_params = [np.copy(param) for param in model.get_params()]
    best_acc = 0.0

    W3_shape = model.W3.shape
    W2_shape = model.W2.shape
    W1_shape = model.W1.shape
    b1_shape = model.b1.shape
    b2_shape = model.b2.shape
    b3_shape = model.b3.shape

    def zero_grads_for_batch():
        gW3 = np.zeros(W3_shape)
        gW2 = np.zeros(W2_shape)
        gW1 = np.zeros(W1_shape)
        gb1 = np.zeros(b1_shape[0])
        gb2 = np.zeros(b2_shape[0])
        gb3 = np.zeros(b3_shape[0])
        return [gW3, gW2, gW1, gb1, gb2, gb3]

    batch_size_modulo = batch_size - 1

    for epoch in xrange(num_epochs):
        t_epoch = time()
        total_loss = 0.0  # total loss in this iteration.
        np.random.shuffle(train_data)

        batch_loss = 0
        batch_grads = zero_grads_for_batch()
        for i, (x, y) in enumerate(train_data):
            loss, grads = model.loss_and_gradients(x, y)

            batch_loss += loss
            batch_grads[0] += grads[0]  # W3
            batch_grads[1] += grads[1]  # W2
            batch_grads[2] += grads[2]  # W1
            batch_grads[3] += grads[3]  # b1
            batch_grads[4] += grads[4]  # b2
            batch_grads[5] += grads[5]  # b3

            if i % batch_size == batch_size_modulo:  # SGD update parameters
                model.W3 -= learning_rate * batch_grads[0]  # W3 update
                model.W2 -= learning_rate * batch_grads[1]  # W2 update
                model.W1 -= learning_rate * batch_grads[2]  # W1 update
                model.b1 -= learning_rate * batch_grads[3]  # b1 update
                model.b2 -= learning_rate * batch_grads[4]  # b2 update
                model.b3 -= learning_rate * batch_grads[5]  # b3 update

                total_loss += batch_loss
                batch_loss = 0
                batch_grads = zero_grads_for_batch()

        if batch_loss != 0:  # there are leftovers from the data that is not in size of batch
            # SGD update parameters
            model.W3 -= learning_rate * batch_grads[0]  # W3 update
            model.W2 -= learning_rate * batch_grads[1]  # W2 update
            model.W1 -= learning_rate * batch_grads[2]  # W1 update
            model.b1 -= learning_rate * batch_grads[3]  # b1 update
            model.b2 -= learning_rate * batch_grads[4]  # b2 update
            model.b3 -= learning_rate * batch_grads[5]  # b3 update

            total_loss += batch_loss

        # notify progress
        train_loss = total_loss / len(train_data)
        dev_accuracy = model.accuracy_on_dataset(dev_data)
        if dev_accuracy > best_acc:
            best_params = [np.copy(param) for param in model.get_params()]
            best_acc = dev_accuracy
        print epoch, 'train_loss:', train_loss, 'time:', time() - t_epoch, 'dev_acc:', dev_accuracy
        log.write('{},{}\n'.format(train_loss, dev_accuracy))

    print 'best accuracy:', best_acc
    model.set_params(best_params)


def train_dev_split(train_x, train_y, size=0.2):
    """
    :param train_x: numpy array of vectors.
    :param train_y: numpy array of integers, each is label associated with train_x.
    :param size: percentage of how much to take from the train data to become dev data.
    """
    train_data = zip(train_x, train_y)
    np.random.shuffle(train_data)
    size = int(len(train_data) * size)

    dev_data = train_data[:size]
    train_data = train_data[size:]
    return train_data, dev_data


def predict_test(test_x, model):
    """
    create a file which contains the prediction of the model on a blind test.
    :param test_x: numpy array of vectors.
    :param model: NN model.
    """
    with open('test.pred', 'w') as f:
        def predict_as_str(x):
            return str(model.predict_on(x))

        preds = map(predict_as_str, test_x)
        f.write('\n'.join(preds))


def main():
    start = time()

    # load data
    print 'loading data'
    mndata = mnist.MNIST('../data')
    train_x, train_y = mndata.load_training()
    train_x = np.array(train_x).astype('float32') / 255

    print 'time to load data:', time() - start
    start = time()

    # set dims
    in_dim = train_x[0].shape[0]
    hid_dim1 = 128
    hid_dim2 = 64
    out_dim = 10

    # create and train classifier
    print 'start training'
    model = MLP1(in_dim, hid_dim1, hid_dim2, out_dim)
    train_data, dev_data = train_dev_split(train_x, train_y, size=0.2)
    print 'all:', len(train_x), ', train:', len(train_data), 'dev:', len(dev_data)
    train_classifier(train_data, dev_data, model)

    # blind test
    print 'start blind test'
    test_x, test_y = mndata.load_testing()
    test_x = np.array(test_x).astype('float32') / 255
    test_acc = model.accuracy_on_dataset(zip(test_x, test_y))
    print 'test-acc:', test_acc

    train_acc = model.accuracy_on_dataset(train_data)
    print 'train-acc:', train_acc

    pickle.dump(model.get_params(), open('model_{}_{}.params'.format(int(train_acc*100), int(test_acc*100)), 'w'))
    log.write('\ntrain-accuracy: {}\ntest-accuracy: {}'.format(train_acc, test_acc))
    log.close()

    print 'time to train:', time() - start


if __name__ == '__main__':
    t0 = time()
    main()
    print 'time to run:', time() - t0

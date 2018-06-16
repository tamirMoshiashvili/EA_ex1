from StringIO import StringIO
from time import time
import mnist
import pickle
from utils import MLP2, np
import sys

log = StringIO()
log.write('train-loss,dev-loss,dev-accuracy\n')


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
        dev_accuracy, dev_loss = model.check_on_dataset(dev_data)
        if dev_accuracy > best_acc:
            best_params = [np.copy(param) for param in model.get_params()]
            best_acc = dev_accuracy
        print epoch, 'time:', time() - t_epoch, \
            'train_loss:', train_loss, 'dev_loss:', dev_loss, 'dev_acc:', dev_accuracy
        log.write('{},{},{}\n'.format(train_loss, dev_loss, dev_accuracy))

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


def main():
    start = time()

    # load data
    print 'loading data'
    mndata = mnist.MNIST(sys.argv[1])
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
    model = MLP2(in_dim, hid_dim1, hid_dim2, out_dim)
    train_data, dev_data = train_dev_split(train_x, train_y, size=0.2)
    print 'all:', len(train_x), ', train:', len(train_data), 'dev:', len(dev_data)
    train_classifier(train_data, dev_data, model)

    # blind test
    print 'start blind test'
    test_x, test_y = mndata.load_testing()
    test_x = np.array(test_x).astype('float32') / 255
    test_acc, test_loss = model.check_on_dataset(zip(test_x, test_y))
    print 'test-acc:', test_acc, 'test-loss:', test_loss

    train_acc, train_loss = model.check_on_dataset(train_data)
    print 'train-acc:', train_acc, 'train-loss:', train_loss

    pickle.dump(model.get_params(), open('nn_params/model_{}_{}.params'.format(int(train_acc * 100), int(test_acc * 100)), 'w'))
    log.write('\ntrain: accuracy: {} | loss: {}\ntest: accuracy: {} | loss: {}'.format(
        train_acc, train_loss, test_acc, test_loss))
    with open('nn_params/log_{}_{}.txt'.format(int(train_acc * 100), int(test_acc * 100)), 'w') as f:
        f.write(log.getvalue())

    print 'time to train:', time() - start


if __name__ == '__main__':
    t0 = time()
    main()
    print 'time to run:', time() - t0

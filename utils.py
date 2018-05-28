import numpy as np


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


def Glorot_init(n, m):
    """
    Xavier Glorot init
    :param n: first dim.
    :param m: second dim.
    :return: numpy array.
    """
    return np.random.uniform(-np.sqrt(6.0 / (n + m)), np.sqrt(6.0 / (n + m)),
                             (n, m) if (n != 1 and m != 1) else n * m)


class MLP2(object):
    def __init__(self, in_dim=784, hid_dim1=100, hid_dim2=50, out_dim=10, params=None):
        if params:
            self.W3 = params[0]
            self.W2 = params[1]
            self.W1 = params[2]
            self.b1 = params[3]
            self.b2 = params[4]
            self.b3 = params[5]
        else:
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

    def loss_and_gradients(self, x, y):
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

    def check_on_dataset(self, dataset):
        """
        :param dataset: list of tuples, each is (x, y) where x is vector and y is its label.
        :return: accuracy and loss of the model on the given dataset, accuracy is float between 0 to 1.
        """
        good = 0.0
        total_loss = 0.0
        for x, y in dataset:
            y_hat = self.forward(x)
            y_prediction = np.argmax(y_hat)
            if y_prediction == y:
                good += 1
            total_loss += -np.log(y_hat[y])
        return good / len(dataset), total_loss / len(dataset)

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

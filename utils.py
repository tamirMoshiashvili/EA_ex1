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
        tan1 = np.tanh(np.dot(x, self.W1) + self.b1)
        tan2 = np.tanh(np.dot(tan1, self.W2) + self.b2)
        return softmax(np.dot(tan2, self.W3) + self.b3)

    def forward_batch(self, batch_x):
        """
        :param batch_x: numpy array of shape (n, in_dim) where n is batch size and in_dim is input dimension.
        :return: numpy array of shape (n, out_dim).
        """
        tan1 = np.tanh(np.dot(batch_x, self.W1) + self.b1)
        tan2 = np.tanh(np.dot(tan1, self.W2) + self.b2)
        batch_out = np.dot(tan2, self.W3) + self.b3
        return np.array([softmax(x) for x in batch_out])

    def check_on_dataset_batch(self, dataset):
        xs, y_golds = zip(*dataset)
        xs = np.array(xs)
        batch_out = self.forward_batch(xs)
        y_preds = np.array([np.argmax(x) for x in batch_out])
        good = float((y_preds == y_golds).sum())
        total_loss = np.array([-np.log(y_hat[y]) for y_hat, y in zip(batch_out, y_golds)]).sum()
        return good / len(y_golds), total_loss / len(y_golds)

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
        :return: loss (float) and gradients (list of size 6).
        """
        tan1 = np.tanh(np.dot(x, self.W1) + self.b1)
        tan2 = np.tanh(np.dot(tan1, self.W2) + self.b2)
        y_hat = softmax(np.dot(tan2, self.W3) + self.b3)
        loss = -np.log(y_hat[y])  # NLL loss

        # gradient of b3
        gb3 = np.copy(y_hat)
        gb3[y] -= 1

        # gradient of W3
        gW3 = np.outer(tan2, y_hat)
        gW3[:, y] -= tan2

        # gradient of b2 - use the chain rule
        dloss_tan2 = -self.W3[:, y] + np.dot(self.W3, y_hat)
        dtan2_db2 = 1 - tan2 ** 2
        gb2 = dloss_tan2 * dtan2_db2

        # gradient of W2 - use the chain rule
        gW2 = np.outer(tan1, gb2)

        # gradient of b1 - use the chain rule
        dtan1_db1 = 1 - tan1 ** 2
        gb1 = np.dot(self.W2, gb2) * dtan1_db1

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

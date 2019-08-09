import numpy as np
import random


class DeepDenseNet:

    def __init__(self, layers, mini_batch_size=0, batch_norm=False, l2_regularization=None, dropout=None):
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.batch_norm = batch_norm
        self.regularization = l2_regularization  # regularization parameter
        self.dropout = dropout  # keep probability
        self.weights = [np.random.randn(x, y) for x, y in zip(layers[:-1], layers[1:])]  # initializing network weights

        # bias acts like beta term in batch_norm since bias is not necessary if batch normalization is true
        self.biases = [np.random.randn(1, x) for x in layers[1:]]
        self.gamma = [np.random.randn(1, x) for x in layers[1:]]  # it is used during batch normalization

        # They are used in adam optimizer
        self.vw = [np.zeros_like(w) for w in self.weights]
        self.sw = [np.zeros_like(w) for w in self.weights]
        self.vb = [np.zeros_like(b) for b in self.biases]
        self.sb = [np.zeros_like(b) for b in self.biases]

        if self.batch_norm:

            self.vg = [np.zeros_like(b) for b in self.biases]
            self.sg = [np.zeros_like(b) for b in self.biases]

        # They are used only during test time
        self.mean = [np.zeros((1, x)) for x in layers[1:]]
        self.variance = [np.ones((1, x)) for x in layers[1:]]

    # Returns output array for given inputs
    def feed_forward(self, inputs):
        a = inputs

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(a, w)
            if self.batch_norm:

                x_hat = (z - self.mean[i]) / (np.sqrt(self.variance[i] + 1e-8))
                z = x_hat * self.gamma[i]
            z += b

            if i < len(self.layers)-2:
                a = self.leaky_relu(z)

            else:
                a = self.softmax(z)

        return a

    def train_network(self, training_data, epoch, learning_rate):

        random.shuffle(training_data)
        mini_batches = [training_data[k:k+self.mini_batch_size] for k in range(0, len(training_data), self.mini_batch_size)]
        for _ in range(epoch):

            for mini_batch in mini_batches:

                x_train, y_train = [np.array(list(x)) for x in zip(*mini_batch)]
                x_train = x_train
                activations = [x_train]
                z_list = []
                a = x_train
                dropout_list = []
                x_hat_list = []
                variance = []
                beta1 = 0.999

                for i, (w, b) in enumerate(zip(self.weights, self.biases)):

                    z = np.dot(a, w)
                    if self.batch_norm:

                        mean = np.mean(z, axis=0, keepdims=True)
                        var = np.var(z, axis=0, keepdims=True)
                        self.mean[i] = beta1*self.mean[i] + (1-beta1)*mean
                        self.variance[i] = beta1*self.variance[i] + (1-beta1)*var
                        variance.append(var)
                        x_hat = (z - mean) / (np.sqrt(var + 1e-8))
                        x_hat_list.append(x_hat)
                        z = x_hat * self.gamma[i]

                    z += b
                    if i < len(self.layers)-2:
                        a = self.leaky_relu(z)
                        if self.dropout:
                            dropout_list.append(np.random.rand(*a.shape) > self.dropout)
                            a[dropout_list[-1]] = 0
                    else:
                        a = self.softmax(z)

                    z_list.append(z)
                    activations.append(a)

                dw = [np.zeros_like(w) for w in self.weights]
                db = [np.zeros_like(b) for b in self.biases]

                if self.batch_norm:

                    dg = [np.zeros_like(b) for b in self.biases]
                dz = activations[-1] - y_train.reshape(y_train.shape[:-1])

                if self.batch_norm:

                    dg[-1], db[-1], dz = self.batch_norm_backward(dz, x_hat_list[-1], variance[-1], 1)

                else:
                    db[-1] = np.sum(dz, axis=0, keepdims=True)/self.mini_batch_size
                dw[-1] = np.dot(activations[-2].T, dz)/self.mini_batch_size

                for i in range(2, len(self.layers)):

                    dz = np.dot(dz, self.weights[-i+1].T) * self.d_leaky_relu(z_list[-i]) * np.invert(dropout_list[-i+1])
                    if self.batch_norm:
                        dg[-i], db[-i], dz = self.batch_norm_backward(dz, x_hat_list[-i], variance[-i], i)
                    else:
                        db[-i] = np.sum(dz, axis=0, keepdims=True)/self.mini_batch_size
                    dw[-i] = np.dot(activations[-i-1].T, dz)/self.mini_batch_size

                if self.regularization:
                    for w in dw:
                        w += (self.regularization/self.mini_batch_size) * w

                if self.batch_norm:
                    self.adam(dw, db, learning_rate, dg)
                else:
                    self.adam(dw, db, learning_rate)

    # adam optimizer
    def adam(self, dw, db, learning_rate, dg=None, beta1=0.9, beta2=0.999, epsilon=1e-8):

        self.vw = [beta1*v + (1-beta1)*w for w, v in zip(dw, self.vw)]
        self.vb = [beta1*v + (1-beta1)*b for b, v in zip(db, self.vb)]
        self.sw = [beta2*s + (1-beta2)*np.square(w) for w, s in zip(dw, self.sw)]
        self.sb = [beta2*s + (1-beta2)*np.square(b) for b, s in zip(db, self.sb)]

        if dg:  # if batch_norm is true

            self.vg = [beta1 * v + (1 - beta1) * g for g, v in zip(dg, self.vg)]
            self.sg = [beta2 * s + (1 - beta2) * np.square(g) for g, s in zip(dg, self.sg)]
            self.gamma = [g - learning_rate*v/np.sqrt(s+epsilon) for g, v, s in zip(self.gamma, self.vg, self.sg)]

        self.weights = [w - learning_rate*v/np.sqrt(s+epsilon) for w, v, s in zip(self.weights, self.vw, self.sw)]
        self.biases = [b - learning_rate*v/np.sqrt(s+epsilon) for b, v, s in zip(self.biases, self.vb, self.sb)]

    # backpropagation through batch normalization layer
    def batch_norm_backward(self, dout, x_hat, variance, index):
        N = dout.shape[0]
        d_gamma = np.sum(dout * x_hat, axis=0, keepdims=True)
        d_beta = np.sum(dout, axis=0, keepdims=True)
        dxhat = dout * self.gamma[-index]
        dz = (N * dxhat - np.sum(dxhat, axis=0, keepdims=True) - x_hat * np.sum(dxhat * x_hat, axis=0, keepdims=True))\
            / (N * np.sqrt(variance + 1e-8))
        return d_gamma, d_beta, dz

    @staticmethod
    def leaky_relu(x):
        return np.maximum(0.001*x, x)

    @staticmethod
    def d_leaky_relu(x):
        r = np.zeros_like(x)
        r[x > 0] = 1
        r[x < 0] = 0.001
        return r

    @staticmethod
    def softmax(x):
        e = np.exp(x - x.max(axis=1, keepdims=True))  # subtracting the max makes it more stable
        return e/np.sum(e, axis=1, keepdims=True)

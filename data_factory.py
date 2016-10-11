import os
import cv2
import numpy as np
import mxnet as mx
from math import sin,cos,sqrt
from sklearn.datasets import fetch_mldata

def onehot_categorical(batch_size, n_labels):
    y = np.zeros((batch_size, n_labels), dtype=np.float32)
    indices = np.random.randint(0, n_labels, batch_size)
    for b in xrange(batch_size):
        y[b, indices[b]] = 1
    return y

def uniform(batch_size, n_dim, n_labels=10, minv=-1, maxv=1, label_indices=None):
    #  z = np.random.uniform(minv, maxv, (batch_size, n_dim)).astype(np.float32)
    def sample(label, n_labels):
        num = int(np.ceil(np.sqrt(n_labels)))
        size = (maxv-minv)*1.0/num
        x, y = np.random.uniform(-size/2, size/2, (2,))
        i = label / num
        j = label % num
        x += j*size+minv+0.5*size
        y += i*size+minv+0.5*size
        return np.array([x, y]).reshape((2,))

    z = np.empty((batch_size, n_dim), dtype=np.float32)
    for batch in xrange(batch_size):
        for zi in xrange(n_dim/2):
            if label_indices is not None:
                z[batch, zi*2:zi*2+2] = sample(label_indices[batch], n_labels)
            else:
                z[batch, zi*2:zi*2+2] = sample(np.random.randint(0, n_labels), n_labels)
    return z

def gaussian(batch_size, n_dim, mean=0, var=1):
    z = np.random.normal(mean, var, (batch_size, n_dim)).astype(np.float32)
    return z

def gaussian_mixture(batch_size, n_dim=2, n_labels=10, x_var=0.5, y_var=0.1, label_indices=None):
    if n_dim % 2 != 0:
        raise Exception("n_dim must be a multiple of 2.")

    def sample(x, y, label, n_labels):
        shift = 1.4
        r = 2.0 * np.pi / float(n_labels) * float(label)
        new_x = x * cos(r) - y * sin(r)
        new_y = x * sin(r) + y * cos(r)
        new_x += shift * cos(r)
        new_y += shift * sin(r)
        return np.array([new_x, new_y]).reshape((2,))

    x = np.random.normal(0, x_var, (batch_size, n_dim / 2))
    y = np.random.normal(0, y_var, (batch_size, n_dim / 2))
    z = np.empty((batch_size, n_dim), dtype=np.float32)
    for batch in xrange(batch_size):
        for zi in xrange(n_dim / 2):
            if label_indices is not None:
                z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], label_indices[batch], n_labels)
            else:
                z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], np.random.randint(0, n_labels), n_labels)

    return z

def swiss_roll(batch_size, n_dim=2, n_labels=10, label_indices=None):
    def sample(label, n_labels):
        uni = np.random.uniform(0.0, 1.0) / float(n_labels) + float(label) / float(n_labels)
        r = sqrt(uni) * 3.0
        rad = np.pi * 4.0 * sqrt(uni)
        x = r * cos(rad)
        y = r * sin(rad)
        return np.array([x, y]).reshape((2,))

    z = np.zeros((batch_size, n_dim), dtype=np.float32)
    for batch in xrange(batch_size):
        for zi in xrange(n_dim / 2):
            if label_indices is not None:
                z[batch, zi*2:zi*2+2] = sample(label_indices[batch], n_labels)
            else:
                z[batch, zi*2:zi*2+2] = sample(np.random.randint(0, n_labels), n_labels)
    return z


class RandIter(mx.io.DataIter):
    def __init__(self, batch_size, n_dim, z_prior='gaussian', n_labels=10, with_label=False, **zargs):
        self.batch_size = batch_size
        self.n_dim = n_dim
        self.provide_data = [('z', (batch_size, n_dim))]
        self.n_labels = n_labels
        if with_label:
            self.provide_label = [('label_n', (batch_size, n_labels))]
            assert z_prior in ['gaussian_mixture', 'swiss_roll', 'uniform']
        else:
            self.provide_label = []
        self.z_prior = z_prior
        self.with_label = with_label
        self.zargs = zargs
        self.tmp_labels = None

    def iter_next(self):
        if self.with_label:
            self.tmp_labels = np.random.randint(0, self.n_labels, (self.batch_size))
        return True

    def getlabel(self):
        if self.with_label:
            label = np.zeros((self.batch_size, self.n_labels), dtype=np.float32)
            for i in xrange(self.batch_size):
                label[i, self.tmp_labels[i]] = 1
            return [mx.nd.array(label)]
        else:
            return []

    def getdata(self):
        if self.with_label:
            self.zargs['label_indices'] = self.tmp_labels

        if self.z_prior == 'gaussian':
            return [mx.nd.array(gaussian(self.batch_size, self.n_dim, **self.zargs))]
        elif self.z_prior == 'uniform':
            return [mx.nd.array(uniform(self.batch_size, self.n_dim, **self.zargs))]
        elif self.z_prior == 'gaussian_mixture':
            return [mx.nd.array(gaussian_mixture(self.batch_size, self.n_dim, n_labels=self.n_labels, **self.zargs))]
        elif self.z_prior == 'swiss_roll':
            return [mx.nd.array(swiss_roll(self.batch_size, self.n_dim, n_labels=self.n_labels, **self.zargs))]
        else:
            raise NotImplementedError


def get_mnist(root_dir='~', train_ratio=0.5, resize=None):
    mnist = fetch_mldata('MNIST original', data_home=os.path.join(root_dir, 'scikit_learn_data'))
    np.random.seed(1234) # set seed for deterministic ordering
    p = np.random.permutation(mnist.data.shape[0])
    X = mnist.data[p]
    X = X.reshape((70000, 28, 28))
    Y = mnist.target[p]

    train_num = int(X.shape[0]*train_ratio)

    if resize:
        X = np.asarray([cv2.resize(x, resize) for x in X])
    X = X.astype(np.float32)/(255.0)
    X = X.reshape((70000, 1, 28, 28))
    #  X = np.tile(X, (1, 3, 1, 1))
    X_train = X[:train_num]
    Y_train = Y[:train_num]
    X_test = X[train_num:]
    Y_test = Y[train_num:]

    return X_train, X_test, Y_train, Y_test

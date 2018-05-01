"""Train LR and generate data."""

import math
import os
import random

import numpy as np
from numpy.linalg import norm

random.seed(42)

k = 3
n = 16
X = np.ones((n, k))
Y = np.ones(n)

w0 = np.arange(float(k))  # np.array([1.0,1.0,0])#np.ones(k)
T = int(1e6)
w_svm = np.array([1.0, 1.0, 0])  # np.array([1 / np.sqrt(2),1 / np.sqrt(2), 0.0])

# Support vectors
# ==================================
X[0] = np.array([0.5, 1.5, 1.0])
X[1] = np.array([-0.5, -1.5, 1.0])
X[2] = np.array([1.5, 0.5, 1.0])
X[3] = np.array([-1.5, -0.5, 1.0])
Y[0] = 1
Y[1] = -1
Y[2] = 1
Y[3] = -1
# ==================================

# linear transformation for creating random data points
M = np.matrix([[1.0, 1.0], [-1.0, 1.0]])
for i in xrange(4, n):
    if i % 2 == 1:
        x_1 = random.uniform(-3.0, -1.0)
        x_2 = random.uniform(-0.5, 0.5)
        X[i][0:2] = M.transpose().dot(np.array([x_1, x_2]))
        Y[i] = -1
    else:
        assert (i % 2 == 0)
        x_1 = random.uniform(1.0, 3.0)
        x_2 = random.uniform(-0.5, 0.5)
        X[i][0:2] = M.transpose().dot(np.array([x_1, x_2]))
        Y[i] = 1


# plt.scatter(X.transpose()[0],X.transpose()[1])


def logistic_regression_loss(w_1, w_2, w_0, X, Y):
    w = [w_1, w_2, w_0]
    res = 0
    for i in xrange(len(Y)):
        res += - np.dot(np.transpose(w), X[i]) * (1 + Y[i]) / 2.0 + np.log(
            1 + np.exp(np.transpose(w).dot(X[i])))
    return res


def gradient_descent(w0, X, Y, T, data_dir, gamma=None):
    """
    Train LR with GD and save data.
    :param w0: initial weights
    :param X: training points
    :param Y: training labels
    :param T: number of GD steps
    :param data_dir: where to save data
    :param gamma: None (without momentum), or gamma value
    """
    w = w0.copy()
    (U, S, V) = np.linalg.svd(X[:, :2])
    # we take the inverse of the max singular value as learning rate.
    eta = 1.0 / S[0]
    print("eta = %.4f" % eta)
    ws = []
    mags = []
    losses = []
    angles = []
    margins = []
    base = math.pow(T, 1 / 1e3)
    t_set = set(int(math.pow(base, i)) for i in xrange(1000))
    t_list = sorted(list(t_set))
    print("base = {}".format(base))
    dw = 0.0
    for t in range(0, T):
        if t in t_set:
            ws.append(w.copy())
            mag = norm(w[:])
            mags.append(mag)
            loss = logistic_regression_loss(w[0], w[1], w[2], X, Y)
            losses.append(loss)
            angle = np.arccos(w_svm[:2].dot(w[:2]) / (norm(w_svm[:2]) * norm(w[:2])))
            angles.append(angle)
            # Correct margin is sqrt(2)
            margin = abs(np.sqrt(2) - np.abs(X[:, :].dot(w[:])).min() / norm(w[:2]))
            margins.append(margin)
            print("[{:d}] t = {:d}, w = {}, mag = {:g}, loss = {:g}, angle = {:g}, margin = {:g}"
                  .format(len(mags), t, w, mag, loss, angle, margin))
        if t % (T // 10) == 0:
            print('{} {}'.format(t, w))
        # grad is d(loss)/dw
        grad = np.zeros(k)
        for i in xrange(n):
            ex = np.exp(X[i].dot(w))
            grad += -(1 + Y[i]) / 2.0 * X[i] + ex / (1 + ex) * X[i]
        # print("grad = ", grad)
        if gamma is None:
            dw = -eta * grad
        else:
            dw = gamma * dw - eta * grad
        w += dw
    out_list = [('t', t_list, '%d'),
                ('w', ws, '%.6e'),
                ('mag', mags, '%.6e'),
                ('loss', losses, '%.6e'),
                ('angle', angles, '%.6e'),
                ('margin', margins, '%.6e')]
    for name, var, fmt in out_list:
        np.savetxt(os.path.join(data_dir, name + '.out'), var, delimiter=',', fmt=fmt)


# def loss_landscape(X, Y):
#     w_1 = np.linspace(-100, 100, 100)
#     w_2 = np.linspace(-100, 100, 100)
#     w_0 = np.ones(1000)
#     W_1, W_2 = np.meshgrid(w_1, w_2)
#     l = logistic_regression_loss(W_1, W_2, 1.0, X, Y)

#     plt.pcolormesh(W_1, W_2, l, cmap='RdBu')
#     plt.colorbar()
#     plt.xlabel(r"$w_1$")
#     plt.ylabel(r"$w_2$")
#     return


# loss_landscape(X,Y)

# RUNS = [('GD', {'gamma': None}),
#         ('GDMO', {'gamma': 0.9})]

RUNS = [('GD', None),
        ('GDMO.9', 0.9),
        ('GDMO.5', 0.5)]


def main():
    def mkdirs(s):
        if not os.path.isdir(s):
            os.makedirs(s)

    os.chdir(os.path.dirname(__file__))
    root_dir = '../data/LR'

    for name, gamma in RUNS:
        data_dir = os.path.join(root_dir, name)
        mkdirs(data_dir)
        print(name)
        gradient_descent(w0, X, Y, T, data_dir, gamma=gamma)


if __name__ == '__main__':
    main()

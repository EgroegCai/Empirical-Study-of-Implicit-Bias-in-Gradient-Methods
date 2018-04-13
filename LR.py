import math
import random
import matplotlib.pyplot as plt
import numpy as np

random.seed(42)

k = 3
n = 16
X = np.ones((n, k))
Y = np.ones(n)
w = np.zeros(k)
T = int(1e3)
w_svm = np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0.0])

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
        res += np.dot(np.transpose(w), X[i]) * (1 + Y[i]) / 2.0 - np.log(
            1 + np.exp(np.transpose(w).dot(X[i])))
    return res


def gradient_descent(w, X, Y, T):
    (U, S, V) = np.linalg.svd(X)
    # we take the inverse of the max singular value as learning rate.
    eta = 1.0 / S[0]
    print("eta = %.4f" % eta)
    angles = []
    mags = []
    loss = []
    base = math.pow(T, 1 / 1e3)
    t_set = set(int(math.pow(base, i)) for i in xrange(1000))
    t_list = sorted(list(t_set))
    print("base = {}".format(base))
    for t in range(1, T + 1):
        if t in t_set:
            loss.append(-logistic_regression_loss(w[0],w[1],w[2],X,Y))
            angle = 1 - np.dot(w_svm, w) / (np.linalg.norm(w_svm) * np.linalg.norm(w))
            angles.append(angle)
            mag = np.linalg.norm(w)
            mags.append(mag)
            print("[{}] t = {}, angle = {:g}, mag = {:g}".format(len(mags), t, angle, mag))
        if t % (T / 10.0) == 0:
            print('{} {}'.format(t, w))
        grad = np.zeros(k)
        for i in xrange(n):
            # print("WX_%d = %.4f" % (i,(w.transpose().dot(X[i]))))
            grad += (1 + Y[i]) / 2 * X[i] - np.exp(w.transpose().dot(X[i])) / (
                    1 + np.exp(w.transpose().dot(X[i]))) * X[i]
        # print("grad = ", grad)
        # we probably need to normalize w to prevent overflow
        w += eta * grad
        # w = w/np.linalg.norm(w,ord=2)
    np.savetxt('data/LR/t.out', t_list, delimiter=',', fmt='%d')
    np.savetxt('data/LR/angle.out', angles, delimiter=',', fmt='%.6e')
    np.savetxt('data/LR/mag.out', mags, delimiter=',', fmt='%.6e')
    np.savetxt('data/LR/loss.out', loss, delimiter=',', fmt='%.6e')
    # print(w)
    return w


def loss_landscape(X, Y):
    w_1 = np.linspace(-100, 100, 100)
    w_2 = np.linspace(-100, 100, 100)
    w_0 = np.ones(1000)
    W_1, W_2 = np.meshgrid(w_1, w_2)
    l = logistic_regression_loss(W_1, W_2, 1.0, X, Y)

    plt.pcolormesh(W_1, W_2, l, cmap='RdBu')
    plt.colorbar()
    plt.xlabel(r"$w_1$")
    plt.ylabel(r"$w_2$")
    return


# loss_landscape(X,Y)


w_star = gradient_descent(w,X,Y,T)
print(w_star)
Xs = np.array([0.1*i - 3 for i in range(60)])
Ys = (-w_star[2] - w_star[0]*Xs)/w_star[1]
plt.plot(Xs,Ys,"--")

index = np.arange(0,len(X)/2,dtype = int)
pos = X[index*2]
neg = X[index*2 + 1]
# plt.plot(ptsXPos,ptsYPos, 'ro')
# plt.plot(ptsXNeg,ptsYNeg, 'bs')
plt.plot(pos.transpose()[0],pos.transpose()[1],'ro')
plt.plot(neg.transpose()[0],neg.transpose()[1],'bs')

Xact = np.array([0.1*i - 3 for i in range(60)])
Yact =  - Xact
plt.plot(Xact,Yact, '-')
plt.axis([-3,3,-3,3])
# plt.axis('equal')

plt.show()

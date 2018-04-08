import scipy as scp
import numpy as np
import random
import matplotlib.pyplot as plt

# random.seed(42)

k = 3
n = 16
X = np.ones((n,k))
Y = np.ones(n)
w = np.ones(k)
T = 1e3

# Support vectors
# ==================================
X[0] = np.array([0.5,1.5,1.0])
X[1] = np.array([1.5,0.5,1.0])
X[2] = np.array([-0.5,-1.5,1.0])
X[3] = np.array([-1.5,-0.5,1.0])
Y[0] = 1
Y[1] = 1
Y[2] = -1
Y[3] = -1
# ==================================

# linear transformation for creating random data points
M = np.matrix([[1.0,1.0],[-1.0,1.0]])
for i in xrange(4,n):
    if i % 2 == 0:
        x_1 = random.uniform(-3.0,-1.0)
        x_2 = random.uniform(-0.5,0.5)
        X[i][0:2] = M.transpose().dot(np.array([x_1,x_2]))
        Y[i] = -1
    else:
        assert(i % 2 == 1)
        x_1 = random.uniform(1.0,3.0)
        x_2 = random.uniform(-0.5,0.5)
        X[i][0:2] = M.transpose().dot(np.array([x_1,x_2]))
        Y[i] = 1

plt.scatter(X.transpose()[0],X.transpose()[1])


def logistic_regression_loss(w,X,Y):
    res = 0
    for i in xrange(len(Y)):
        res += np.dot(np.transpose(w),X[i])*Y[i]
    return res 

def gradient_descent(w,X,Y,T):
    (U,S,V) = np.linalg.svd(X)
    # we take the inverse of the max singular value as learning rate.
    eta = 1.0/(S[0])
    print("eta = %.4f" % eta)
    for t in xrange(int(T)):
        if t % (int(T)/10) == 0: print('{} {}'.format(t,w))
        temp = np.zeros(k)
        for i in xrange(n):
            # print("WX_%d = %.4f" % (i,(w.transpose().dot(X[i]))))
            temp += (1+Y[i])/2*X[i]-np.exp(w.transpose().dot(X[i]))/(1+np.exp(w.transpose().dot(X[i])))*X[i]
        # print("temp = ",temp)
        # we probabily need to normalize w to prevent overflow
        w += eta*temp
        # w = w/np.linalg.norm(w,ord=2)
    print(w)
    return w


w_star = gradient_descent(w,X,Y,T)
Xs = np.array([0.1*i - 3 for i in range(60)])
Ys = (-w_star[2] - w_star[0]*Xs)/w_star[1]
plt.plot(Xs,Ys)

Xact = np.array([0.1*i - 3 for i in range(60)])
Yact =  - Xact
plt.plot(Xact,Yact, 'g^')

plt.show()        


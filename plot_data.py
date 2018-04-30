import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
from LR import X, Y

axis_font = {'fontname':'Arial', 'size':'20'}


steps = np.loadtxt("data/LR/t.out", delimiter=',')
ws = np.loadtxt("data/LR/w.out", delimiter=',')
mags = np.loadtxt("data/LR/mag.out", delimiter=',')
loss = np.loadtxt("data/LR/loss.out", delimiter=',')
angles = np.loadtxt("data/LR/angle.out", delimiter=',')
margin = np.loadtxt("data/LR/margin.out", delimiter=',')

w_star = ws[-1, :]


def plot_w(w):
    print(w)
    # plot SVM boundary
    Xs = np.array([0.1 * i - 3 for i in range(60)])
    Ys = (-w[2] - w[0] * Xs) / w[1]
    plt.plot(Xs, Ys, "--",label="SVM")

    # plot data points X
    index = np.arange(0, len(X) / 2, dtype=int)
    pos = X[index * 2]
    neg = X[index * 2 + 1]
    # plt.plot(ptsXPos,ptsYPos, 'ro')
    # plt.plot(ptsXNeg,ptsYNeg, 'bs')
    plt.plot(pos.transpose()[0], pos.transpose()[1], 'ro')
    plt.plot(neg.transpose()[0], neg.transpose()[1], 'bs')

    # plot LR decision boundary
    Xact = np.array([0.1 * i - 3 for i in range(60)])
    Yact = - Xact
    plt.plot(Xact, Yact, '-',label="LR")

    plt.axis([-3, 3, -3, 3])
    plt.gca().set_aspect('equal')
    plt.legend(loc="upper left")
    plt.title("Logistic Regression on linearly-separable dataset")
    plt.xlabel(r"$x_1$",**axis_font)
    plt.ylabel(r"$x_2$",**axis_font)
    plt.savefig("data/LR/LR_sample_data_point.png")
    plt.show()



def plot_angle_vs_step(angles, steps):
    plt.figure()
    print(np.shape(steps))
    print(np.shape(angles))
    plt.semilogx(steps,angles)
    plt.title(r"Angles between $w$ and $w_{svm}$")
    plt.xlabel(r"$t$",**axis_font)
    plt.ylabel(r"$cos^{-1}(\langle w,w_{svm}\rangle/(\|w\|\cdot\|w_{svm}\|))$",**axis_font)
    plt.savefig("data/LR/Angles.png")
    plt.show()

def plot_mag_vs_step(mags,steps):
    plt.figure()
    mags = mags/mags[-1]
    plt.semilogx(steps,mags)
    plt.title(r"Norm of $w$ versus $t$")
    plt.xlabel(r"$t$",**axis_font)
    plt.ylabel(r"Normalized $\|w(t)\|$",**axis_font)
    plt.savefig("data/LR/Norm_vs_time.png")
    plt.show()

def plot_loss_vs_step(loss,steps):
    plt.figure()
    plt.loglog(steps,loss)
    plt.title(r"$L(w(t))$ vs $t$")
    plt.xlabel(r"$t$",**axis_font)
    plt.ylabel(r"$L(w(t))$",**axis_font)
    plt.savefig("data/LR/loss_vs_time.png")
    plt.show()

def plot_margin_vs_step(margin,steps):
    plt.figure()
    plt.loglog(steps,margin)
    plt.title(r"Margin Gap vs $t$")
    plt.xlabel(r"$t$",**axis_font)
    plt.ylabel(r"Margin Gap",**axis_font)
    plt.savefig("data/LR/margin_vs_time.png")
    plt.show()

plot_w(w_star)
plot_angle_vs_step(angles,steps)
plot_mag_vs_step(mags,steps)
plot_loss_vs_step(loss,steps)
plot_margin_vs_step(margin,steps)

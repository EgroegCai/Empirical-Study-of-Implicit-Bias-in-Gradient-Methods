import numpy as np
import scipy as scp 
import matplotlib.pyplot as plt

steps = np.loadtxt("data/LR/t.out",delimiter=',')
angles = np.loadtxt("data/LR/angle.out",delimiter=',')
loss = np.loadtxt("data/LR/loss.out",delimiter=',')
# margin = np.loadtxt("data/LR/margin.out",delimiter=',')
mags = np.loadtxt("data/LR/mag.out",delimiter=',')


def plot_angle_vs_step(angles,steps):
    fig1 = plt.figure()
    print(np.shape(steps))
    print(np.shape(angles))
    plt.semilogx(steps,angles)
    plt.title(r"Angles between $w$ and $w_{svm}$")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$\frac{\langle w,w_{svm}\rangle}{\|w\|\|w_{svm}\|}$")
    plt.savefig("data/LR/Angles.png")
    plt.show()
    return

def plot_mag_vs_step(mags,steps):
    fig2 = plt.figure()
    mags = mags/mags[-1]
    plt.semilogx(steps,mags)
    plt.title(r"Norm of $w$ versus t")
    plt.xlabel(r"$t$")
    plt.ylabel(r"Normalized $\|w(t)\|$")
    plt.savefig("data/LR/Norm_vs_time.png")
    plt.show()
    return

def plot_loss_vs_step(loss,steps):
    fig3 = plt.figure()
    plt.loglog(steps,loss)
    plt.title(r"$L(w(t))$ vs $t$")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$L(w(t))$")
    plt.savefig("data/LR/loss_vs_time.png")
    plt.show()
    return

def plot_margin_vs_step(margin,steps):
    fig4 = plt.figure()
    plt.semilogx(steps,margin)
    plt.title(r"Margin Gap versus t")
    plt.xlabel(r"$t$")
    plt.ylabel(r"Margin Gap")
    plt.savefig("data/LR/margin_vs_time.png")
    plt.show()
    return

plot_angle_vs_step(angles,steps)
plot_mag_vs_step(mags,steps)
plot_loss_vs_step(loss,steps)
# plot_margin_vs_step(margin,steps)

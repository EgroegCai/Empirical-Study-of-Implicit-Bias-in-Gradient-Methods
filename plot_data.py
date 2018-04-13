import numpy as np
import scipy as scp 
import matplotlib.pyplot as plt

steps = np.loadtxt("data/t.out",delimiter=',')
angles = np.loadtxt("data/angle.out",delimiter=',')
mags = np.loadtxt("data/mag.out",delimiter=',')


def plot_angle_vs_step(angles,steps):
    fig1 = plt.figure()
    print(np.shape(steps))
    print(np.shape(angles))
    plt.semilogx(steps,angles)
    plt.title(r"Angles between current decision boundary and max-margin decision boundary")
    plt.xlabel("T")
    plt.ylabel(r"$\frac{\langle w,w_{svm}\rangle}{\|w\|\|w_{svm}\|}$")
    plt.savefig("Angles.png")
    plt.show()
    return

def plot_mag_vs_step(mag,steps):
    fig2 = plt.figure()
    plt.semilogx(steps,mags)
    plt.title(r"Norm of $w$ versus steps")
    plt.savefig("Norm_vs_time.png")
    plt.savefig("mag.png")
    plt.show()
    return

plot_angle_vs_step(angles,steps)
plot_mag_vs_step(mags,steps)

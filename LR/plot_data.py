import os

import matplotlib.pyplot as plt
import numpy as np

from LR import X, RUNS

axis_font = {'fontname': 'Arial', 'size': '20'}


def main():
    show_fig = True
    # show_fig = False
    os.chdir(os.path.dirname(__file__))
    root_dir = '../data/LR'
    labels = [r[0] for r in RUNS]
    n_runs = len(labels)
    # List of dict
    data_list = [{} for _ in range(n_runs)]
    var_list = ['t', 'w', 'mag', 'loss', 'angle', 'margin']

    def load_data(d, datadir):
        """Load data from file into dict"""
        for s in var_list:
            d[s] = np.loadtxt(os.path.join(datadir, s + '.out'), delimiter=',')

    for run_name, d in zip(labels, data_list):
        load_data(d, os.path.join(root_dir, run_name))
        # Normalize mag
        d['mag'] /= d['mag'][-1]

    w_stars = [d['w'][-1, :] for d in data_list]
    plot_w(w_stars, labels, os.path.join(root_dir, 'LR_sample_data_point.png'), show=show_fig)

    def get_series(k):
        return [(d['t'], d[k]) for d in data_list]

    plot_series(get_series('angle'), labels, r"Angles between $w$ and $w_{svm}$",
                r"$t$", r"$cos^{-1}(\langle w,w_{svm}\rangle/(\|w\|\cdot\|w_{svm}\|))$",
                plt.semilogx, os.path.join(root_dir, 'Angles.png'), show=show_fig)

    plot_series(get_series('mag'), labels, r"Norm of $w$ versus $t$",
                r"$t$", r"Normalized $\|w(t)\|$",
                plt.semilogx, os.path.join(root_dir, 'Norm_vs_time.png'), show=show_fig)

    plot_series(get_series('loss'), labels, r"$L(w(t))$ vs $t$",
                r"$t$", r"$L(w(t))$",
                plt.loglog, os.path.join(root_dir, 'loss_vs_time.png'), show=show_fig)

    plot_series(get_series('margin'), labels, r"Margin Gap vs $t$",
                r"$t$", r"Margin Gap",
                plt.loglog, os.path.join(root_dir, 'margin_vs_time.png'), show=show_fig)

    # plot_angle_vs_step(angles, steps)
    # plot_mag_vs_step(mags, steps)
    # plot_loss_vs_step(loss, steps)
    # plot_margin_vs_step(margin, steps)


def plot_series(data_list, labels, title, xlab, ylab, plot_fn=plt.plot,
                fname=None, show=True, label_kwargs=axis_font):
    """Plot multiple time series."""
    plt.figure()
    for (x, y), label in zip(data_list, labels):
        plot_fn(x, y, label=label)
    plt.legend(loc="best")
    plt.title(title)
    plt.xlabel(xlab, **label_kwargs)
    plt.ylabel(ylab, **label_kwargs)
    if fname:
        plt.savefig(fname)
    if show:
        plt.show()


def plot_w(w_list, labels, fname=None, show=True):
    print(w_list)
    # plot LR boundary
    for w, label in zip(w_list, labels):
        Xs = np.linspace(-3, 3, 50)
        Ys = (-w[2] - w[0] * Xs) / w[1]
        plt.plot(Xs, Ys, "-", label=label)

    # plot data points X
    index = np.arange(0, len(X) / 2, dtype=int)
    pos = X[index * 2]
    neg = X[index * 2 + 1]
    # plt.plot(ptsXPos,ptsYPos, 'ro')
    # plt.plot(ptsXNeg,ptsYNeg, 'bs')
    plt.plot(pos.transpose()[0], pos.transpose()[1], 'ro')
    plt.plot(neg.transpose()[0], neg.transpose()[1], 'bs')

    # plot SVM decision boundary
    Xact = np.linspace(-3, 3, 50)
    Yact = - Xact
    plt.plot(Xact, Yact, '--', label="SVM")

    plt.axis([-3, 3, -3, 3])
    plt.gca().set_aspect('equal')
    plt.legend(loc="best")
    plt.title("Logistic Regression on linearly-separable dataset")
    plt.xlabel(r"$x_1$", **axis_font)
    plt.ylabel(r"$x_2$", **axis_font)
    if fname:
        plt.savefig(fname)
    if show:
        plt.show()


# def plot_angle_vs_step(angles, steps):
#     plt.figure()
#     print(np.shape(steps))
#     print(np.shape(angles))
#     plt.semilogx(steps,angles)
#     plt.title(r"Angles between $w$ and $w_{svm}$")
#     plt.xlabel(r"$t$",**axis_font)
#     plt.ylabel(r"$cos^{-1}(\langle w,w_{svm}\rangle/(\|w\|\cdot\|w_{svm}\|))$",**axis_font)
#     plt.savefig("data/LR/Angles.png")
#     plt.show()
#
# def plot_mag_vs_step(mags,steps):
#     plt.figure()
#     mags = mags/mags[-1]
#     plt.semilogx(steps,mags)
#     plt.title(r"Norm of $w$ versus $t$")
#     plt.xlabel(r"$t$",**axis_font)
#     plt.ylabel(r"Normalized $\|w(t)\|$",**axis_font)
#     plt.savefig("data/LR/Norm_vs_time.png")
#     plt.show()
#
# def plot_loss_vs_step(loss,steps):
#     plt.figure()
#     plt.loglog(steps,loss)
#     plt.title(r"$L(w(t))$ vs $t$")
#     plt.xlabel(r"$t$",**axis_font)
#     plt.ylabel(r"$L(w(t))$",**axis_font)
#     plt.savefig("data/LR/loss_vs_time.png")
#     plt.show()
#
# def plot_margin_vs_step(margin,steps):
#     plt.figure()
#     plt.loglog(steps,margin)
#     plt.title(r"Margin Gap vs $t$")
#     plt.xlabel(r"$t$",**axis_font)
#     plt.ylabel(r"Margin Gap",**axis_font)
#     plt.savefig("data/LR/margin_vs_time.png")
#     plt.show()


if __name__ == '__main__':
    main()

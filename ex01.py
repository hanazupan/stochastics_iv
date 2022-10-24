import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from  matplotlib.animation import FuncAnimation
import seaborn as sns

sns.color_palette("hls", 8)


# ASK: should the first t be added? else not normalised
def gaussian(x, t, D=0.1):
    return 1/np.sqrt(4*np.pi*D*t)*np.exp(-x**2/(4*D*t))


def plot_gaussian_in_time(D=0.1):
    xs = np.linspace(-3, 3)
    ts = np.linspace(0.05, 40, 40)

    def func(frame, *fargs):
        ax.clear()
        ax.set_ylim(0, 1)
        ax.set_xlabel("x")
        ax.set_ylabel("probability")
        line = ax.plot(xs, gaussian(xs, ts[frame], D=D), c="black")
        ax.text(1.8, 0.9, f"t={round(ts[frame])}\nD={D}")
        return line,

    fig, ax = plt.subplots()
    ax.set_ylim(0, 1)
    anim = FuncAnimation(fig, func, frames=len(ts), interval=200)
    anim.save('analytic_ex01.mp4', fps=3)
    plt.show()


def brownian_motion(tau=1, N_steps=100, x0=0, D=0.1):
    trajectory = [x0]
    x_current = x0
    for k in range(N_steps-1):
        x_next = x_current + np.sqrt(2*D*tau)*np.random.normal()
        trajectory.append(x_next)
        x_current = x_next
    return trajectory


def plot_trajectory(length=10000, how_many=5):
    times = np.arange(length)
    fig, ax = plt.subplots()
    for i in range(how_many):
        traj = brownian_motion(N_steps=length)
        ax.plot(times, traj)
    ax.set_xlabel("t/tau")
    ax.set_ylabel("position")
    plt.savefig("brownian_motion", dpi=500)
    plt.show()


def trajectory_histograms(length=5000, how_many=100):
    all_traj = np.zeros(shape=(how_many, length))
    for i in range(how_many):
        traj = brownian_motion(N_steps=length)
        all_traj[i] = traj
    bin_num = 30
    all_histos = np.zeros((bin_num, length))
    for i in range(length):
        histo, bins = np.histogram(all_traj[:, i], bins=bin_num, range=[-3, 3])
        all_histos[:, i] = histo
    return all_histos, bins


def plot_histogram_in_time(H, bins):
    xs = np.linspace(-3, 3)

    def func(frame, *fargs):
        ax.clear()
        ax.set_ylim(0, 1)
        ax.set_xlabel("x")
        ax.set_ylabel("probability")
        bar = plt.bar(bins[:-1], H[:, frame], width=1)
        ax.text(1.8, 0.9, f"step={frame}")
        return bar,

    fig, ax = plt.subplots()
    ax.set_ylim(0, 1)
    anim = FuncAnimation(fig, func, frames=H.shape[1], interval=200)
    anim.save('histogram_ex01.mp4', fps=3)
    plt.show()


if __name__ == "__main__":
    # plot_gaussian_in_time()
    # plot_trajectory()
    H, bins = trajectory_histograms()
    plot_histogram_in_time(H, bins)
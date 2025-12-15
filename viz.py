import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


# Loss and gradient
def f(w):
    return w[0] ** 2 + 10 * w[1] ** 2


def grad(w):
    return np.array([2 * w[0], 20 * w[1]])


# Optimizers
def sgd(w, g, lr):
    return w - lr * g


def momentum(w, g, v, lr, gamma=0.9):
    v = gamma * v + lr * g
    return w - v, v


# Optimize
def run(start, opt="sgd", lr=0.1, steps=50):
    w, path, v = np.array(start, float), [np.array(start, float)], np.zeros(2)
    for _ in range(steps):
        g = grad(w)
        w, v = momentum(w, g, v, lr) if opt == "momentum" else (sgd(w, g, lr), v)
        path.append(w.copy())
    return np.array(path)


# Visualize
def viz(start=(4, 2), lr=0.1, steps=50):
    os.makedirs("viz", exist_ok=True)

    sgd_p = run(start, "sgd", lr, steps)
    mom_p = run(start, "momentum", lr, steps)

    w1, w2 = np.linspace(-5, 5, 100), np.linspace(-3, 3, 100)
    W1, W2 = np.meshgrid(w1, w2)
    Z = f([W1, W2])

    fig = plt.figure(figsize=(16, 10))

    # 3D plots
    for i, (path, name, col) in enumerate(
        [(sgd_p, "SGD", "r"), (mom_p, "Momentum", "b")], 1
    ):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        ax.plot_surface(W1, W2, Z, cmap="viridis", alpha=0.6)
        loss = [f(p) for p in path]
        ax.plot(path[:, 0], path[:, 1], loss, f"{col}.-", lw=2, ms=6)
        ax.scatter(*start, f(start), c="red", s=100)
        ax.scatter(0, 0, 0, c="green", s=100, marker="*")
        ax.set_title(f"{name} 3D", fontweight="bold")
        ax.view_init(25, 45)

    # 2D contours
    for i, (path, name, col) in enumerate(
        [(sgd_p, "SGD", "r"), (mom_p, "Momentum", "b")], 3
    ):
        ax = fig.add_subplot(2, 2, i)
        ax.contour(W1, W2, Z, 20, cmap="viridis", alpha=0.6)
        ax.plot(path[:, 0], path[:, 1], f"{col}.-", lw=2, ms=6, label=name)
        ax.scatter(*start, c="red", s=100, zorder=5)
        ax.scatter(0, 0, c="green", s=150, marker="*", zorder=5)
        g = grad(np.array(start))
        gn = -g / np.linalg.norm(g) * 0.5
        ax.arrow(start[0], start[1], gn[0], gn[1], head_width=0.2, fc="orange", lw=2)
        ax.set_title(f"{name} Contour", fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(f"viz/opt_lr{lr}.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSGD final: {sgd_p[-1]}, loss: {f(sgd_p[-1]):.6f}")
    print(f"Momentum final: {mom_p[-1]}, loss: {f(mom_p[-1]):.6f}")


viz(lr=0.1)
viz(lr=0.2)

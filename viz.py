# viz.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from optimiser import f, grad, run


def viz(
    start=(4, 2), lr=0.1, steps=50, optimizers=["sgd", "momentum", "rmsprop", "adam"]
):
    os.makedirs("viz", exist_ok=True)

    paths = {opt: run(start, opt, lr, steps) for opt in optimizers}
    colors = {"sgd": "r", "momentum": "b", "rmsprop": "g", "adam": "m"}

    w1, w2 = np.linspace(-5, 5, 100), np.linspace(-3, 3, 100)
    W1, W2 = np.meshgrid(w1, w2)
    Z = f([W1, W2])

    n = len(optimizers)
    fig = plt.figure(figsize=(12, 6 * n))

    for i, opt in enumerate(optimizers, 1):
        path, col = paths[opt], colors[opt]

        # 3D
        ax = fig.add_subplot(n, 2, 2 * i - 1, projection="3d")
        ax.plot_surface(W1, W2, Z, cmap="viridis", alpha=0.5)
        loss = [f(p) for p in path]
        ax.plot(path[:, 0], path[:, 1], loss, f"{col}.-", lw=2, ms=5)
        ax.scatter(*start, f(start), c="red", s=100)
        ax.scatter(0, 0, 0, c="lime", s=100, marker="*")
        ax.set_title(f"{opt.upper()} 3D", fontweight="bold")
        ax.view_init(25, 45)

        # 2D
        ax = fig.add_subplot(n, 2, 2 * i)
        ax.contour(W1, W2, Z, 20, cmap="viridis", alpha=0.5)
        ax.plot(path[:, 0], path[:, 1], f"{col}.-", lw=2, ms=5, label=opt.upper())
        ax.scatter(*start, c="red", s=100, zorder=5)
        ax.scatter(0, 0, c="lime", s=150, marker="*", zorder=5)
        g = grad(np.array(start))
        gn = -g / np.linalg.norm(g) * 0.5
        ax.arrow(start[0], start[1], gn[0], gn[1], head_width=0.2, fc="orange", lw=2)
        ax.set_title(f"{opt.upper()} Contour", fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_aspect("equal")

    plt.tight_layout()
    fname = f"viz/{'_'.join(optimizers)}_lr{lr}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n{'='*50}\nLR: {lr}, Steps: {steps}\n{'='*50}")
    for opt in optimizers:
        path = paths[opt]
        print(f"{opt.upper():10s} final: {path[-1]}, loss: {f(path[-1]):.6f}")

    return paths


if __name__ == "__main__":
    viz(lr=0.1, steps=50)

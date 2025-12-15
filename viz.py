# viz.py - UPDATE
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from optimiser import f_simple, f_rosenbrock, grad_simple, grad_rosenbrock, run


def viz(
    start=(4, 2),
    lr=0.1,
    steps=50,
    optimizers=["sgd", "momentum", "rmsprop", "adam"],
    loss_fn="simple",
):
    os.makedirs("viz", exist_ok=True)

    # Select functions based on loss_fn
    if loss_fn == "simple":
        f, grad_fn = f_simple, grad_simple
        w1_range, w2_range = (-5, 5), (-3, 3)
    elif loss_fn == "rosenbrock":
        f, grad_fn = f_rosenbrock, grad_rosenbrock
        w1_range, w2_range = (-2, 2), (-1, 3)

    paths = {opt: run(start, opt, lr, steps, loss_fn=loss_fn) for opt in optimizers}
    colors = {"sgd": "r", "momentum": "b", "rmsprop": "g", "adam": "m"}

    w1, w2 = np.linspace(*w1_range, 100), np.linspace(*w2_range, 100)
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
        ax.plot(path[:, 0], path[:, 1], loss, f"{col}.-", lw=2, ms=4)
        ax.scatter(*start, f(start), c="red", s=100)
        target = (1, 1) if loss_fn == "rosenbrock" else (0, 0)
        ax.scatter(*target, f(target), c="lime", s=100, marker="*")
        ax.set_title(f"{opt.upper()} 3D - {loss_fn}", fontweight="bold")
        ax.view_init(25, 45)

        # 2D
        ax = fig.add_subplot(n, 2, 2 * i)
        ax.contour(W1, W2, Z, 30, cmap="viridis", alpha=0.5)
        ax.plot(path[:, 0], path[:, 1], f"{col}.-", lw=2, ms=4, label=opt.upper())
        ax.scatter(*start, c="red", s=100, zorder=5)
        ax.scatter(*target, c="lime", s=150, marker="*", zorder=5)
        g = grad_fn(np.array(start))
        gn = -g / np.linalg.norm(g) * 0.3
        ax.arrow(start[0], start[1], gn[0], gn[1], head_width=0.1, fc="orange", lw=2)
        ax.set_title(f"{opt.upper()} Contour - {loss_fn}", fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fname = f"viz/{loss_fn}_{'_'.join(optimizers)}_lr{lr}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n{'='*60}\n{loss_fn.upper()} - LR: {lr}, Steps: {steps}\n{'='*60}")
    for opt in optimizers:
        path = paths[opt]
        print(f"{opt.upper():10s} final: {path[-1]}, loss: {f(path[-1]):.6f}")

    return paths


if __name__ == "__main__":
    print("SIMPLE BOWL:")
    viz(start=(4, 2), lr=0.1, steps=50, loss_fn="simple")

    print("\n\nROSENBROCK lr=0.1:")
    viz(start=(-1.5, 2.5), lr=0.001, steps=1000, loss_fn="rosenbrock")

# optimizers.py
import numpy as np


def f(w):
    return w[0] ** 2 + 10 * w[1] ** 2


def grad(w):
    return np.array([2 * w[0], 20 * w[1]])


def sgd(w, g, state, lr):
    return w - lr * g, None


def momentum(w, g, state, lr, gamma=0.9):
    v = state if state is not None else np.zeros_like(w)
    v = gamma * v + lr * g
    return w - v, v


def rmsprop(w, g, state, lr, beta=0.9, eps=1e-8):
    s = state if state is not None else np.zeros_like(w)
    s = beta * s + (1 - beta) * g**2
    return w - lr * g / (np.sqrt(s) + eps), s


def run(start, opt="sgd", lr=0.1, steps=50, **kwargs):
    w, path, state = np.array(start, float), [np.array(start, float)], None
    opt_fn = {"sgd": sgd, "momentum": momentum, "rmsprop": rmsprop}[opt]

    for _ in range(steps):
        g = grad(w)
        w, state = opt_fn(w, g, state, lr, **kwargs)
        path.append(w.copy())
    return np.array(path)

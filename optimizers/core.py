import numpy as np


def f_simple(w):
    return w[0] ** 2 + 10 * w[1] ** 2


def grad_simple(w):
    return np.array([2 * w[0], 20 * w[1]])


def f_rosenbrock(w, a=1, b=100):
    return (a - w[0]) ** 2 + b * (w[1] - w[0] ** 2) ** 2


def grad_rosenbrock(w, a=1, b=100):
    dw1 = -2 * (a - w[0]) - 400 * w[0] * (w[1] - w[0] ** 2)
    dw2 = 200 * (w[1] - w[0] ** 2)
    return np.array([dw1, dw2])


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


def adam(w, g, state, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    if state is None:
        m, v, t = np.zeros_like(w), np.zeros_like(w), 0
    else:
        m, v, t = state

    t += 1
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g**2

    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)

    w_new = w - lr * m_hat / (np.sqrt(v_hat) + eps)
    return w_new, (m, v, t)


def run(start, opt="sgd", lr=0.01, steps=50, loss_fn="simple", **kwargs):
    w, path, state = np.array(start, float), [np.array(start, float)], None
    opt_fn = {"sgd": sgd, "momentum": momentum, "rmsprop": rmsprop, "adam": adam}[opt]

    if loss_fn == "simple":
        grad_fn = grad_simple
    elif loss_fn == "rosenbrock":
        grad_fn = grad_rosenbrock
    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")

    for _ in range(steps):
        g = grad_fn(w)
        w, state = opt_fn(w, g, state, lr, **kwargs)
        path.append(w.copy())
    return np.array(path)

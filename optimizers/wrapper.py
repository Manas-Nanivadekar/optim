import torch
import numpy as np

from .core import sgd, momentum, rmsprop, adam


class CustomOptimizer:
    def __init__(self, params, opt_type="sgd", lr=0.001, **kwargs):
        self.params = list(params)
        self.opt_type = opt_type
        self.lr = lr
        self.kwargs = kwargs
        self.state = {id(p): None for p in self.params}

        self.opt_fn = {
            "sgd": sgd,
            "momentum": momentum,
            "rmsprop": rmsprop,
            "adam": adam,
        }[opt_type]

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue

            w = p.data.cpu().numpy().flatten()
            g = p.grad.data.cpu().numpy().flatten()

            state = self.state[id(p)]

            w_new, state = self.opt_fn(w, g, state, self.lr, **self.kwargs)

            self.state[id(p)] = state
            p.data = torch.from_numpy(w_new.reshape(p.data.shape)).to(p.device)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

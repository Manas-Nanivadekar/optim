import torch
import numpy as np
from .core import sgd, momentum, rmsprop, adam


class CustomOptimizer:
    """PyTorch adapter for custom optimizers"""

    def __init__(self, params, opt_type="sgd", lr=0.001, weight_decay=0.0, **kwargs):
        self.params = list(params)
        self.opt_type = opt_type
        self.lr = lr
        self.weight_decay = weight_decay
        self.kwargs = kwargs

        self.state = {id(p): None for p in self.params}

        self.opt_fn = {
            "sgd": sgd,
            "momentum": momentum,
            "rmsprop": rmsprop,
            "adam": adam,
        }[opt_type]

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue

            w = p.data.cpu().numpy().flatten()
            g = p.grad.data.cpu().numpy().flatten()

            if self.weight_decay > 0:
                g = g + self.weight_decay * w

            state = self.state[id(p)]

            w_new, state = self.opt_fn(w, g, state, self.lr, **self.kwargs)

            self.state[id(p)] = state
            p.data = torch.from_numpy(w_new.reshape(p.data.shape)).to(p.device)

    def state_dict(self):
        state_dict = {
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "opt_type": self.opt_type,
            "kwargs": self.kwargs,
            "param_states": [],
        }

        for i, p in enumerate(self.params):
            param_state = self.state[id(p)]

            if param_state is not None and isinstance(param_state, np.ndarray):
                param_state = param_state.tolist()
            elif param_state is not None and isinstance(param_state, tuple):

                param_state = tuple(
                    x.tolist() if isinstance(x, np.ndarray) else x for x in param_state
                )
            state_dict["param_states"].append(param_state)

        return state_dict

    def load_state_dict(self, state_dict):
        self.lr = state_dict["lr"]
        self.weight_decay = state_dict["weight_decay"]
        self.opt_type = state_dict["opt_type"]
        self.kwargs = state_dict["kwargs"]

        for i, p in enumerate(self.params):
            if i < len(state_dict["param_states"]):
                param_state = state_dict["param_states"][i]

                if param_state is not None and isinstance(param_state, list):
                    param_state = np.array(param_state)
                elif param_state is not None and isinstance(param_state, tuple):
                    param_state = tuple(
                        np.array(x) if isinstance(x, list) else x for x in param_state
                    )

                self.state[id(p)] = param_state

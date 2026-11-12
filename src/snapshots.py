"""Save and temporarily activate trainable parameters without copying base weights."""
from contextlib import contextmanager

import torch


class TrainableSnapshot:
    def __init__(self, model, version):
        self.version = version
        self.state = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        self.bytes = sum(v.numel() * v.element_size() for v in self.state.values())

    @contextmanager
    def activate(self, model):
        parameters = dict(model.named_parameters())
        candidate = {n: parameters[n].detach().clone() for n in self.state}
        try:
            with torch.no_grad():
                for name, value in self.state.items():
                    parameters[name].copy_(value)
            yield
        finally:
            with torch.no_grad():
                for name, value in candidate.items():
                    parameters[name].copy_(value)

    def restore(self, model):
        parameters = dict(model.named_parameters())
        with torch.no_grad():
            for name, value in self.state.items():
                parameters[name].copy_(value)

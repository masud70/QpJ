import copy
import warnings

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.flop_counter import flop_registry

from data.celeba_loader import to_float


class FlopCounter(TorchDispatchMode):
    """PyTorch's FLOP formulas without module tracking, which breaks on double-backward (gradient penalty)."""

    total = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        formula = flop_registry.get(func._overloadpacket)
        if formula:
            self.total += formula(*args, **kwargs, out_val=out)
        return out


def count_flops(fn):
    with FlopCounter() as counter:
        fn()
    return counter.total


def flops_per_image(method, sampler):
    if hasattr(method, "diffusion"):
        x = torch.randn(1, 3, method.size, method.size, device=method.device)
        t = torch.zeros(1, dtype=torch.long, device=method.device)
        return count_flops(lambda: method.ema(x, t)) * method.network_calls(sampler)
    return count_flops(lambda: method.sample(1, sampler))


def capacity(method, images, batch_size):
    """Params and FLOPs (matmul/conv, incl. backward and GP double-backward) measured on a throwaway copy."""
    probe = copy.deepcopy(method)
    batch = to_float(images[:batch_size].to(method.device))
    probe.set_mode(True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_flops = count_flops(lambda: probe.train_step(lambda: batch))
    probe.set_mode(False)
    probe.prepare_sampling(images[:max(batch_size, 256)])
    params = {name: sum(p.numel() for p in net.parameters()) for name, net in method.networks().items()}
    return {
        "params": {"total": sum(params.values()), "per_network": params},
        "flops_per_train_step": train_flops,
        "flops_per_image": {s: flops_per_image(probe, s) for s in method.samplers},
    }

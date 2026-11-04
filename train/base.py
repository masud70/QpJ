import torch


def adam(params, c):
    return torch.optim.Adam(params, lr=c["lr"], betas=tuple(c["betas"]))


def step(optimizer, loss):
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


class Method:
    """Common interface: train_step consumes batches via next_batch(); sample returns images in [-1, 1]."""

    samplers = ["direct"]
    eval_sampler = "direct"

    def __init__(self, cfg, device):
        self.cfg, self.device = cfg, device
        self.size = cfg["data"]["image_size"]
        self.latent = cfg["model"]["latent_dim"]

    def networks(self):
        raise NotImplementedError

    def train_step(self, next_batch):
        raise NotImplementedError

    def sample(self, n, sampler):
        raise NotImplementedError

    def network_calls(self, sampler):
        return 1

    def prepare_sampling(self, images):
        pass

    def reconstruct(self, x):
        return None

    def set_mode(self, training):
        for net in self.networks().values():
            net.train(training)

    def lr(self):
        return self.optimizers[0].param_groups[0]["lr"]

    def state(self):
        return {name: net.state_dict() for name, net in self.networks().items()}

    def noise(self, n):
        return torch.randn(n, self.latent, device=self.device)

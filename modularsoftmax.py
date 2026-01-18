import torch
from torch.optim import Optimizer

"""
Steepest descent

TODO: Nesterov momentum
"""

class ModularSoftmaxOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, eps=1e-8):
        defaults = dict(lr=lr, eps=eps)
        super(ModularSoftmaxOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                if p.ndim == 2:
                    G_rho = p.grad[1:]
                    norms = torch.norm(G_rho, p=2, dim=1, keepdim=True)
                    G_rho.div_(norms.add_(group['eps']))

                    d_o = p.shape[0]
                    delta_theta_0 = G_rho.sum(dim=0).mul_(-1.0 / d_o)

                    p.add_(delta_theta_0, alpha=group['lr'])
                    p[1:].add_(G_rho, alpha=group['lr'])
                elif p.ndim == 1:
                    G_rho = p.grad[1:]
                    G_rho.sign_()

                    d_o = p.shape[0]
                    delta_theta_0 = G_rho.sum(dim=0).mul_(-1.0 / d_o)

                    p.add_(delta_theta_0, alpha=group['lr'])
                    p[1:].add_(G_rho, alpha=group['lr'])
                else:
                    raise ValueError(f"{p.ndim=}")

        return loss

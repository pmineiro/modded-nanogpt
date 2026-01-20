import torch
import torch.distributed as dist
from torch.optim import Optimizer

"""
Steepest descent

TODO: Nesterov momentum
"""

class MaxRowModularOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, eps=1e-8):
        defaults = dict(lr=lr, eps=eps)
        super(MaxRowModularOptimizer, self).__init__(params, defaults)

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

                if p.ndim == 1 or p.ndim == 2:
                    G_rho = p.grad[1:]

                    if p.ndim == 1:
                        G_rho.sign_()
                    else:
                        norms = torch.norm(G_rho, p=2, dim=1, keepdim=True)
                        G_rho.div_(norms.add_(group['eps']))

                    d_o = p.shape[0]
                    delta_theta_0 = G_rho.sum(dim=0).mul_(-1.0 / d_o)

                    p.add_(delta_theta_0, alpha=group['lr'])
                    p[1:].add_(G_rho, alpha=group['lr'])
                else:
                    raise ValueError(f"{p.ndim=}")

        return loss

class DistMaxRowModular:
    def __init__(self, params, lr=3e-4, eps=1e-6):
        defaults = dict(lr=lr, eps=eps)
        self.param_groups = [{'params': list(params), **defaults}]
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self._reduce_scatter_hooks = []
        self._reduce_scatter_futures = {}
        self.should_sync = False

        for p in params:
            self._reduce_scatter_hooks.append(
                p.register_post_accumulate_grad_hook(self._sync_gradient)
            )

    def _sync_gradient(self, param):
        if not self.should_sync:
            return

        grad = param.grad
        if param.numel() < 1024:
            fut = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True)
            self._reduce_scatter_futures[param] = (fut, grad)
        else:
            rank_size = grad.shape[0] // self.world_size
            grad_slice = torch.empty_like(grad[:rank_size])
            fut = dist.reduce_scatter_tensor(
                grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True
            )
            self._reduce_scatter_futures[param] = (fut, grad_slice)

    @torch.no_grad()
    def step(self):
        all_gather_futures = []

        for param in self.param_groups[0]['params']:
            if param not in self._reduce_scatter_futures:
                continue

            fut, g_slice = self._reduce_scatter_futures[param]
            fut.wait()  # wait for averaged gradient slice

            lr = self.param_groups[0]['lr']
            eps = self.param_groups[0]['eps']

            is_large = param.numel() >= 1024
            if is_large:
                rank_size = param.shape[0] // self.world_size
                local_start = self.rank * rank_size
                local_end   = (self.rank + 1) * rank_size
                p_local = param[local_start:local_end]      # local view, shape [rank_size, hidden]
            else:
                p_local = param                             # full param for small ones

            if p_local.ndim == 1 or p_local.ndim == 2:
                local_has_row0 = (local_start == 0) if is_large else True

                if local_has_row0:
                    G_rho_local = p_local.grad[1:] if not is_large else g_slice[1:]
                else:
                    G_rho_local = g_slice if is_large else p_local.grad

                if p_local.ndim == 1:
                    G_rho_local.sign_()
                else:
                    norms = torch.norm(G_rho_local, p=2, dim=1, keepdim=True)
                    G_rho_local.div_(norms.add_(eps))

                sum_G_rho_local = G_rho_local.sum(dim=0)
                dist.all_reduce(sum_G_rho_local, op=dist.ReduceOp.SUM)

                global_d_o = param.shape[0]   # full vocab size, same on all ranks
                delta_theta_0 = sum_G_rho_local.mul_(-1.0 / global_d_o)

                p_local.add_(delta_theta_0, alpha=lr) # all rows get this

                if local_has_row0:
                    p_local[1:].add_(G_rho_local, alpha=lr)
                else:
                    p_local.add_(G_rho_local, alpha=lr)
            else:
                raise ValueError(f"{p_local.ndim=}")

            if is_large:
                fut = dist.all_gather_into_tensor(param, p_local, async_op=True)
                all_gather_futures.append(fut.get_future())

        if all_gather_futures:
            torch.futures.collect_all(all_gather_futures).wait()

        self._reduce_scatter_futures.clear()

    def zero_grad(self, set_to_none=True):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()

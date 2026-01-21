import torch
import torch.distributed as dist
from torch.optim import Optimizer

class MaxRowModularOptimizer(Optimizer):
    """
    Implements steepest descent under the (∞, 2) modular norm for softmax-based policies.

    This optimizer addresses the extra degree of freedom in softmax parameterization by
    working in relative coordinates (ρ) and enforcing a zero-sum constraint on updates
    (∑Δθ = 0).

    The update rule follows the "modular norm":
    1. Gradients are normalized row-wise by their L2 norm, motivating a change in
       each logit by a maximal unit step.
    2. A shift (Δθ₀) is applied to all rows to ensure the total update is centered,
       removing redundant drift in the parameter space.

    Args:
        params (iterable): iterable of parameters to optimize
        lr (float): learning rate (default: 1e-3)
        eps (float): term added to the denominator to improve numerical stability (default: 1e-8)
    """

    def __init__(self, params, lr=1e-3, momentum=0, eps=1e-8, centered=True):
        if not (0 <= momentum < 1):
            raise ValueError(f"{momentum=}")

        defaults = dict(lr=lr, momentum=momentum, eps=eps)
        super(MaxRowModularOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue

                if p.ndim == 1 or p.ndim == 2:
                    d_p = p.grad

                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                        else:
                            buf = param_state['momentum_buffer']
                            # Standard momentum update: v = mu * v + g
                            buf.mul_(momentum).add_(d_p)

                        # Nesterov update: g = g + mu * v
                        d_p.add_(buf, alpha=momentum)

                    if self.centered:
                        G_rho = d_p[1:]
                    else:
                        G_rho = d_p

                    if p.ndim == 1:
                        G_rho.sign_()
                    else:
                        norms = torch.norm(G_rho, p=2, dim=1, keepdim=True)
                        G_rho.div_(norms.add_(group['eps']))

                    if self.centered:
                        d_o = p.shape[0]
                        delta_theta_0 = G_rho.sum(dim=0).mul_(-1.0 / d_o)

                        p.sub_(delta_theta_0, alpha=group['lr'])
                        p[1:].sub_(G_rho, alpha=group['lr'])
                    else:
                        p.sub_(G_rho, alpha=group['lr'])
                else:
                    raise ValueError(f"{p.ndim=}")

        return loss

class DistMaxRowModular:
    """
    Distributed version of MaxRowModularOptimizer with Nesterov Momentum.
    Shards momentum buffers to save memory on large parameters.
    """
    def __init__(self, params, lr=3e-4, momentum=0.0, eps=1e-6):
        if not (0 <= momentum < 1):
            raise ValueError(f"Invalid momentum: {momentum}")

        defaults = dict(lr=lr, momentum=momentum, eps=eps)
        self.param_groups = [{'params': list(params), **defaults}]
        self.state = {} # To store momentum buffers

        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self._reduce_scatter_hooks = []
        self._reduce_scatter_futures = {}
        self.should_sync = False

        for p in self.param_groups[0]['params']:
            self.state[p] = {} # Initialize state for each param
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
        group = self.param_groups[0]
        lr, momentum, eps = group['lr'], group['momentum'], group['eps']

        for param in group['params']:
            if param not in self._reduce_scatter_futures:
                continue

            fut, g_slice = self._reduce_scatter_futures[param]
            fut.wait()  # Wait for averaged/scattered gradient

            # --- 1. Momentum Logic ---
            if momentum != 0:
                param_state = self.state[param]
                if 'momentum_buffer' not in param_state:
                    # Initialize sharded buffer
                    param_state['momentum_buffer'] = torch.clone(g_slice).detach()
                    buf = param_state['momentum_buffer']
                else:
                    buf = param_state['momentum_buffer']
                    # v = mu * v + g
                    buf.mul_(momentum).add_(g_slice)

                # Nesterov: g = g + mu * v
                g_slice.add_(buf, alpha=momentum)

            # --- 2. Sharding / Slicing Logic ---
            is_large = param.numel() >= 1024
            if is_large:
                rank_size = param.shape[0] // self.world_size
                local_start = self.rank * rank_size
                p_local = param[local_start : local_start + rank_size]
                local_has_row0 = (local_start == 0)
            else:
                p_local = param
                local_has_row0 = True

            # --- 3. Modular Norm Logic ---
            if p_local.ndim == 1 or p_local.ndim == 2:
                # G_rho calculation
                if local_has_row0:
                    G_rho_local = g_slice[1:]
                else:
                    G_rho_local = g_slice

                # Steepest Descent Step
                if p_local.ndim == 1:
                    G_rho_local.sign_()
                else:
                    norms = torch.norm(G_rho_local, p=2, dim=1, keepdim=True)
                    G_rho_local.div_(norms.add_(eps))

                # --- 4. Zero-Sum Constraint (Global) ---
                sum_G_rho_local = G_rho_local.sum(dim=0)
                # Need the global sum of all G_rho across all ranks
                dist.all_reduce(sum_G_rho_local, op=dist.ReduceOp.SUM)

                global_d_o = param.shape[0]
                delta_theta_0 = sum_G_rho_local.mul_(-1.0 / global_d_o)

                # --- 5. Apply Updates ---
                p_local.sub_(delta_theta_0, alpha=lr)
                if local_has_row0:
                    p_local[1:].sub_(G_rho_local, alpha=lr)
                else:
                    p_local.sub_(G_rho_local, alpha=lr)
            else:
                raise ValueError(f"{p_local.ndim=}")

            # --- 6. Synchronize Shards ---
            if is_large:
                fut = dist.all_gather_into_tensor(param, p_local, async_op=True)
                all_gather_futures.append(fut.get_future())

        if all_gather_futures:
            torch.futures.collect_all(all_gather_futures).wait()

        self._reduce_scatter_futures.clear()

    def zero_grad(self, set_to_none=True):
        """
        Clears the gradients of all optimized parameters.
        set_to_none=True is recommended for lower memory footprint.
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()

        # Crucial for your distributed implementation:
        # Ensure any leftover tracking from the previous step is wiped.
        self._reduce_scatter_futures.clear()

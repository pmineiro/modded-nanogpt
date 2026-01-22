import math
import torch

@torch.compile(fullgraph=True)
def compute_malbo_parameters(pi, alpha=0.05):
    """
    Computes v_hat, kappa, and gamma using vectorized bisection on the GPU.
    """
    B, T = pi.shape

    # 2. Bounds for v (Eqn in App 2.2 and 2.3)
    # Lower bound is a small fraction of the max reward
    # Upper bound is the empirical mean
    v_high = pi.mean(dim=1)
    v_low = (alpha * math.exp(-1) / (T * (1 + T))) * pi.max(dim=1).values
    v = (v_low + v_high) / 2.0

    # Target wealth threshold: log((1+T)/alpha)
    target_log_wealth = math.log((1.0 + T) / alpha)

    if pi.dtype == torch.float64:
        n_iters = 50
    elif pi.dtype == torch.float32:
        n_iters = 23
    else:
        n_iters = 16 # note: bfloat16 numerically unstable, don't do this

    # 3. Nested Bisection
    # Outer loop finds v, inner loop finds optimal bet b* for that v
    for _ in range(n_iters): # Outer bisection (v)
        b_low = torch.zeros(B, device=pi.device, dtype=pi.dtype)
        b_high = torch.ones(B, device=pi.device, dtype=pi.dtype)
        b = 0.5 * torch.ones(B, device=pi.device, dtype=pi.dtype)

        # Inner bisection to find b* that maximizes wealth for current v
        # We find the root of the derivative of log wealth w.r.t b
        for _ in range(n_iters):
            # f'(b) = sum( (r - v) / (v + b(r - v)) )
            reldiff = pi / v.unsqueeze(1) - 1
            denom = 1 + b.unsqueeze(1) * reldiff
            grad_b = (reldiff / denom).sum(dim=1)

            b_low = torch.where(grad_b > 0, b, b_low)
            b_high = torch.where(grad_b <= 0, b, b_high)
            b = (b_low + b_high) / 2.0

        # Calculate log wealth at b*
        # log K = sum( log(1 + b*(r/v - 1)) )
        log_wealth = torch.log1p(b.unsqueeze(1) * (pi / v.unsqueeze(1) - 1.0)).sum(dim=1)

        # Update v bisection
        # If wealth > target, v is too small (increase v)
        v_low = torch.where(log_wealth > target_log_wealth, v, v_low)
        v_high = torch.where(log_wealth <= target_log_wealth, v, v_high)
        v = (v_low + v_high) / 2.0

    # 4. Compute Kappa and Normalize
    rel = pi / v.unsqueeze(1)
    kappa = rel / (1 + b.unsqueeze(1) * (rel - 1))
    kappa = kappa / (kappa.sum(dim=1, keepdim=True) + 1e-8)

    return v, kappa

if __name__ == "__main__":
    def test():
        import numpy as np
        from sanitycheck import vhat as vhat_sanity_check

        B, T, K = 10, 128, 1000
        device = "cuda" if torch.cuda.is_available() else "cpu"

        alpha = 0.05
        logits = 4 * torch.randn(B, T, K, device=device, dtype=torch.float32)
        targets = torch.randint(0, K, (B, T), device=device)
        log_probs = torch.log_softmax(logits, dim=-1)
        log_pi = torch.gather(log_probs, -1, targets.unsqueeze(-1)).squeeze(-1)
        pi = log_pi.exp()

        vhat, kappa = compute_malbo_parameters(pi, alpha=alpha)
        vhat_numpy = vhat.float().cpu().numpy()
        kappa_numpy = kappa.float().cpu().numpy()

        r_numpy = pi.float().cpu().numpy()
        assert np.all(r_numpy > 0), np.min(r_numpy)

        for n in range(B):
            rn_np = r_numpy[n]
            vhat_sanity, bstar_sanity = vhat_sanity_check(wrs=rn_np, alpha=alpha)
            rel_np = rn_np / vhat_sanity
            kappa_sanity = rel_np / (1 + bstar_sanity * (rel_np - 1))
            kappa_sanity /= (np.sum(kappa_sanity) + 1e-8)

            worst_kappa_idx = np.argmax(np.abs(kappa_sanity - kappa_numpy[n]))
            min_kappa = np.min(kappa_numpy[n])
            max_kappa = np.max(kappa_numpy[n])

            assert np.allclose(vhat_sanity, vhat_numpy[n], atol=1e-5), f"{vhat_sanity=} {vhat_numpy[n]=}"
            assert np.allclose(kappa_sanity, kappa_numpy[n], atol=1e-3), f"{worst_kappa_idx=} {kappa_sanity[worst_kappa_idx]=} {kappa_numpy[n,worst_kappa_idx]=} {min_kappa=} {max_kappa=} {bstar_sanity=} {vhat_sanity=}"

        print("✅ MALBO parameter sanity check passed!")

    test()

from math import log
from scipy.optimize import brentq
import numpy as np
import numpy.typing as npt

def log_wealth(*, wrs: npt.NDArray[np.number], v: float, lam: float):
    return np.sum(np.log1p(lam * (wrs/v - 1)))

def grad_log_wealth(*, wrs: npt.NDArray[np.number], v: float, lam: float):
    return np.sum((wrs - v) / (v + lam * (wrs - v)))

def lambda_star(*, wrs: npt.NDArray[np.number], v: float):
    """
    note log_wealth is negative definite (strictly negative hessian)
    """

    grad_log_wealth_lower = grad_log_wealth(wrs=wrs, v=v, lam=0)
    if grad_log_wealth_lower <= 0:
        return 0

    maxlam = len(wrs) / (1 + len(wrs))
    grad_log_wealth_upper = grad_log_wealth(wrs=wrs, v=v, lam=maxlam)
    if grad_log_wealth_upper >= 0:
        return maxlam

    return brentq(lambda z: grad_log_wealth(wrs=wrs, v=v, lam=z), 0, maxlam, xtol=1e-6, rtol=1e-6)

def log_wealth_star(*, wrs: npt.NDArray[np.number], v: float):
    lamstar = lambda_star(wrs=wrs, v=v)
    return log_wealth(wrs=wrs, v=v, lam=lamstar)

def vhat(*, wrs: npt.NDArray, alpha: float):
    """
    vhat = \max{ v \in [0, 1] : log_wealth_star(wrs=wrs, v=vhat) >= log(1/alpha) \}

    log_wealth_star is decreasing in v:

    let v_1 < v_2, then
    log_wealth_star(v_2) = log_wealth(b^*(v_2), v_2)
                          <= log_wealth(b^*(v_2), v_1)                (decreasing in v)
                          <= log_wealth(b^*(v_1), v_1)                (optimality of b^*)
                          = log_wealth_star(v_1)
    """
    vemp = np.mean(wrs).item()

    if vemp <= 0:
        return 0

    T = len(wrs)
    vmin = (alpha / (np.exp(1).item() * T * (1 + T))) * np.max(wrs).item()

    threshold = log((1+T)/alpha)
    log_wealth_zero = log_wealth_star(wrs=wrs, v=vmin)
    if log_wealth_zero < threshold:
        assert False
        return vmin

    log_wealth_vemp = log_wealth_star(wrs=wrs, v=vemp)
    if log_wealth_vemp >= threshold:
        assert False
        return vemp

    def objective(v):
        return log_wealth_star(wrs=wrs, v=v) - threshold

    v = brentq(objective, vmin, vemp, xtol=1e-6)
    lamstar = lambda_star(wrs=wrs, v=v)
    return v, lamstar

if __name__ == "__main__":
    import scipy.stats
    import time
    from tqdm import trange

    with np.errstate(invalid='raise'):
        alpha = 0.05
        n_wrs = 100000
        n_sims = 100

        times = []
        vhats = []
        fails = []
        wmaxs = []
        wmeans = []
        kappamaxs = []
        kappameans = []

        # importance weights pareto distributed with mean 1 = scale * b / (b - 1)
        b = 10/9
        scale = (b - 1) / b
        print(f"Using pareto distribution with {b=} and {scale=}")

        # rewards independent and binomial with rate 0.5

        for _ in trange(n_sims):
            """
            1. generate a sample of wrs which is pareto distributed with a mean of 0.5
            2. compute vhat <-- time how long it takes to do this
            """
            ws = scipy.stats.pareto.rvs(b=b, scale=scale, size=n_wrs)
            rs = scipy.stats.binom.rvs(p=0.5, n=1, size=n_wrs)
            wrs = ws * rs

            start_time = time.time()
            v = vhat(wrs=wrs, alpha=alpha)
            if v > 0:
                lamstar = lambda_star(wrs=wrs, v=v)
                kappa = lamstar * wrs / (v + lamstar * (wrs - v))
            else:
                lamstar = 0
                kappa = 0
            end_time = time.time()

            times.append(end_time - start_time)
            vhats.append(v)
            fails.append(1 if v > 0.5 else 0)
            wmaxs.append(np.max(wrs))
            wmeans.append(np.mean(ws))
            kappamaxs.append(np.max(kappa))
            kappameans.append(np.median(kappa))

        print(f"Average time per vhat computation: {np.mean(times):.4f} +/- {3*np.std(times):.4f} seconds")
        print(f"vhat: [{np.min(vhats):.4f}, {np.mean(vhats):.4f}, {np.max(vhats):.4f}]")
        print(f"Empirical mis-coverage: {np.mean(fails):.4f}")
        print(f"Median wmax: {np.median(wmaxs):.4f}")
        print(f"Mean wmean: {np.mean(wmeans):.4f}")
        print(f"Median kappamax: {np.median(kappamaxs):.4f}")
        print(f"Mean kappamean: {np.mean(kappameans):.4f}")

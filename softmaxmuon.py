import torch

@torch.compile(fullgraph=True)
def mean_center(X: torch.tensor) -> torch.Tensor:
    assert X.ndim == 1 or X.ndim == 2

    """
    return (I - n^{-1} \vec{1} \vec{1}^\top) X
    """

    return X - X.mean(dim=0)

@torch.compile(fullgraph=True)
def hdagger_x(p: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    assert p.ndim == 1
    assert X.ndim == 1 or X.ndim == 2

    # Tikhonov
    eps = 1e-3 / p.size(-1)
    inv_p = p / (p**2 + eps**2)

    if X.ndim == 2:
        inv_p = inv_p.unsqueeze(1)

    return mean_center(inv_p * X)

@torch.compile(fullgraph=True)
def inverse_sqrt_ns(K: torch.Tensor, *, iterations: int = 5) -> torch.Tensor:
    assert K.ndim == 2

    """
    Precondition K

    alpha = \|K\|_F
    """

    """
    Compute x^{-1/2} using coupled Newton-Schulz iteration

    Y_0 = K / alpha
    Z_0 = I

    T_k = 1/2 (3 I - Z_k Y_k)
    Y_{k+1} = Y_k T_k
    Z_{k+1} = T_k Z_k

    after T iterations

    return Z_T / sqrt(alpha)
    """

    d = K.shape[0]
    alpha = torch.norm(K, 'fro')
    Y = K / alpha
    I = torch.eye(d, device=K.device, dtype=K.dtype)
    Z = I.clone()

    # Use a fixed number of iterations (e.g., 20 for sufficient convergence)
    for _ in range(iterations):
        T = 0.5 * (3 * I - Z @ Y)
        Y = Y @ T
        Z = T @ Z

    return Z / torch.sqrt(alpha)

@torch.compile(fullgraph=True)
def softmax_muon(logits: torch.Tensor, G: torch.Tensor, *, epsilon: float = 1e-3) -> torch.Tensor:
    assert logits.ndim >= 2 # batch x ... x vocab
    assert G.ndim == 2

    p = torch.softmax(logits, dim=-1)
    p_bar = p.view(-1, p.size(-1)).mean(dim=0)
    tildeG = mean_center(G)
    B = hdagger_x(p_bar, tildeG)        # TODO: this operation needs higher numerical precision
    K = tildeG.t() @ B
    K = (K + K.T) / 2
    sqrtK = inverse_sqrt_ns(K)          # TODO: perturb K to avoid ill-conditioning ... add a multiple of the identity (?)
    W = B @ sqrtK

    return W

if __name__ == "__main__":
    def test_hdagger():
        """
        randomized unit test for hdagger_x

        let p be a point in the $d$-dimensional probability simplex
            1^\top p = 1, p > 0.

        let H = Diag(p) - p p^\top

        let x be an arbitrary vector

        let z = mean_center(x)
        let y = hdagger_x(p, z)

        then hdagger_x is correct if

        H y = z and 1^\top y = 0
        """

        torch.manual_seed(42)
        device = 'cpu'
        dtype = torch.float64
        num_tests = 10

        for i in range(num_tests):
            d = torch.randint(5, 50, (1,)).item()  # Random dimension between 5 and 50
            p = torch.rand(d, device=device, dtype=dtype)
            p = p / p.sum()  # Normalize to probability simplex
            x = torch.randn(d, device=device, dtype=dtype)
            z = mean_center(x)
            y = hdagger_x(p, z)
            H = torch.diag(p) - torch.outer(p, p)
            H_y = H @ y
            recon_error = torch.norm(H_y - z) / torch.norm(z + 1e-10)  # Avoid div by zero if z near zero
            sum_y = y.sum().abs()
            #print(f"Test {i+1}: Dim {d}, Recon error: {recon_error.item():.2e}, Sum y: {sum_y.item():.2e}")
            assert recon_error.item() < 1e-3, f"Recon error {recon_error.item():.2e} exceeds tolerance"
            assert sum_y.item() < 1e-4, f"Sum y {sum_y.item():.2e} exceeds tolerance"

        print("test_hdagger: All tests passed!")

    test_hdagger()

    def test_inverse_sqrt_ns():
        """
        randomized unit test for inverse_sqrt_ns

        compare inverse_sqrt_ns to torch.linalg.sqrtm
        using randomly generated symmetric positive-definite matrices
        choosing a high number of iterations (e.g., 10) to ensure inverse_sqrt_ns "should" converge
        """

        def generate_spd(d: int, device: str = 'cpu', dtype: torch.dtype = torch.float64) -> torch.Tensor:
            A = torch.randn(d, d, device=device, dtype=dtype)
            K = A.T @ A
            deltaK = torch.trace(K) / d
            K += 1e-2 * deltaK * torch.eye(d, device=device, dtype=dtype) # Ensure positive definite
            return K

        torch.manual_seed(42)
        device = 'cpu'
        dtype = torch.float32
        num_tests = 10

        for i in range(num_tests):
            d = torch.randint(5, 50, (1,)).item()  # Random dimension between 5 and 50
            K = generate_spd(d, device, dtype)
            L, Q = torch.linalg.eigh(K)
            inv_sqrt_L = 1.0 / torch.sqrt(L)
            ref = Q @ torch.diag_embed(inv_sqrt_L) @ Q.T
            approx = inverse_sqrt_ns(K, iterations=15)
            error = torch.norm(ref - approx) / torch.norm(ref)
            #print(f"Test {i+1}: Dim {d}, Relative error: {error.item():.2e}")
            assert error.item() < 1e-4, f"Max relative error {error.item():.2e} exceeds tolerance"

        print("test_inverse_sqrt_ns: All tests passed!")

    test_inverse_sqrt_ns()

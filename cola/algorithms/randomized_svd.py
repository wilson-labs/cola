from cola.ops import LinearOperator
from cola.utils import export


@export
def randomized_svd(A: LinearOperator, rank: int):
    """
    Computes the randomized SVD decomposition of the linear operator A,
    A = U Sigma V^*

    Args:
        A (LinearOperator): A linear operator of size (n, n).
        rank (int): The rank of the SVD approximation (r,).

    Returns:
        tuple:
            - Sigma (Array): A vector of size (r,), containing the singular values.
            - U (Array): Left matrix of the decomposition of size (n, r).
            - V (Array): Right matrix of the decomposition of size (n, r).

    """
    xnp = A.ops
    Omega = xnp.randn(*(A.shape[0], rank), dtype=A.dtype)
    Y = A @ Omega
    Q, _ = xnp.qr(Y, full_matrices=False)
    B = Q.T @ A
    U, Sigma, V = xnp.svd(B, full_matrices=False)
    return Sigma, Q @ U, V

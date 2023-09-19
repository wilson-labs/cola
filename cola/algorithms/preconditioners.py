from typing import Union
from cola.ops import LinearOperator
from cola.algorithms import power_iteration
# from plum import dispatch
from cola.utils import export
# from cola.compositions import UnitaryDecomposition


@export
class NystromPrecond(LinearOperator):
    """
    Constructs the Nystrom Preconditioner of a linear operator A.

    Args:
        A (LinearOperator): A positive definite linear operator of size (n, n).
        rank (int): The rank of the Nystrom approximation.
        mu (float): Regularization of the linear system (A + mu)x = b.
         Usually, this preconditioner is used to solve linear systems and
         therefore its construction accomodates for the regularization.
        eps (float): Shift used when constructing the preconditioner.
        adjust_mu (bool, optional): Whether to adjust the regularization with the
         estimatted dominant eigenvalue.
    """
    def __init__(self, A, rank, mu=1e-7, eps=1e-8, adjust_mu=True, key=None):
        super().__init__(dtype=A.dtype, shape=A.shape)
        key = A.xnp.PNRGKey(42) if key is None else key
        Omega = A.xnp.randn(*(A.shape[0], rank), dtype=A.dtype, device=A.device, key=key)
        Lambda, U = get_nys_approx(A=A, Omega=Omega, eps=eps)
        self.adjusted_mu = amu = mu * A.xnp.max(self.Lambda, axis=0) if adjust_mu else mu
        # Num and denom help for defining inverse and sqrt
        subspace_num = A.xnp.min(Lambda) + amu
        subspace_denom = Lambda + amu
        subspace_scaling = subspace_num / subspace_denom - 1
        subspace_scaling = subspace_scaling[:, None]
        preconditioned_eigmax = A.xnp.min(Lambda) + amu
        preconditioned_eigmin = amu
        self.U = U
        self.subspace_scaling = subspace_scaling

    def _matmat(self, V):
        subspace_term = self.U @ (self.subspace_scaling * (self.U.T @ V))
        return subspace_term + V


@export
class NystromPrecond(LinearOperator):
    """
    Constructs the Nystrom Preconditioner of a linear operator A.

    Args:
        A (LinearOperator): A positive definite linear operator of size (n, n).
        rank (int): The rank of the Nystrom approximation.
        mu (float): Regularization of the linear system (A + mu)x = b.
         Usually, this preconditioner is used to solve linear systems and
         therefore its construction accomodates for the regularization.
        eps (float): Shift used when constructing the preconditioner.
        adjust_mu (bool, optional): Whether to adjust the regularization with the
         estimatted dominant eigenvalue.

    Returns:
        LinearOperator: Nystrom Preconditioner.
    """
    def __init__(self, A, rank, mu=1e-7, eps=1e-8, adjust_mu=True, key=None):
        super().__init__(dtype=A.dtype, shape=A.shape)
        key = self.xnp.PNRGKey(42) if key is None else key
        Omega = self.xnp.randn(*(A.shape[0], rank), dtype=A.dtype, device=A.device, key=key)
        self._create_approx(A=A, Omega=Omega, mu=mu, eps=eps, adjust_mu=adjust_mu)

    def _create_approx(self, A, Omega, mu, eps, adjust_mu):
        xnp = self.xnp
        self.Lambda, self.U = get_nys_approx(A=A, Omega=Omega, eps=eps)
        self.adjusted_mu = amu = mu * xnp.max(self.Lambda, axis=0) if adjust_mu else mu
        # Num and denom help for defining inverse and sqrt
        self.subspace_num = xnp.min(self.Lambda) + amu
        self.subspace_denom = self.Lambda + amu
        self.subspace_scaling = self.subspace_num / self.subspace_denom - 1
        self.subspace_scaling = self.subspace_scaling[:, None]
        self.preconditioned_eigmax = xnp.min(self.Lambda) + amu
        self.preconditioned_eigmin = amu

    def _matmat(self, V):
        subspace_term = self.U @ (self.subspace_scaling * (self.U.T @ V))
        return subspace_term + V


def get_nys_approx(A, Omega, eps):
    xnp = A.xnp
    Omega, _ = xnp.qr(Omega, full_matrices=False)
    Y = A @ Omega
    nu = eps * xnp.norm(Y, ord="fro")
    Y += nu * Omega
    C = xnp.cholesky(Omega.T @ Y)
    aux = xnp.solvetri(C, Y.T, lower=True)
    B = aux.T  # shape (params, rank)
    U, Sigma, _ = xnp.svd(B, full_matrices=False)
    Lambda = xnp.clip(Sigma**2.0 - nu, a_min=0.0)
    return Lambda, U


# @dispatch
# def sqrt(A: Union[NystromPrecond, NystromPrecondLazy]) -> NystromPrecondLazy:
#     xnp = A.xnp
#     subspace_num = xnp.sqrt(xnp.copy(A.subspace_num))
#     subspace_denom = xnp.sqrt(xnp.copy(A.subspace_denom))
#     B = NystromPrecondLazy(A.dtype, A.shape, xnp.copy(A.U), subspace_num, subspace_denom)
#     return B

# @dispatch
# def inv(A: Union[NystromPrecond, NystromPrecondLazy]) -> NystromPrecondLazy:
#     xnp = A.xnp
#     subspace_num, subspace_denom = xnp.copy(A.subspace_denom), xnp.copy(A.subspace_num)
#     B = NystromPrecondLazy(A.dtype, A.shape, xnp.copy(A.U), subspace_num, subspace_denom)
#     return B


class AdaNysPrecond(LinearOperator):
    def __init__(self, A, rank, bounds, mult=1.5, mu=1e-7, eps=1e-8, adjust_mu=True):
        super().__init__(dtype=A.dtype, shape=A.shape)
        xnp = A.xnp
        Omega = xnp.randn(*(A.shape[0], rank), dtype=A.dtype, device=A.device)
        Lambda, self.U = get_nys_approx(A=A, Omega=Omega, eps=eps)
        self.error = estimate_approx_error(A, Lambda, self.U, tol=1e-7, max_iter=1000)
        i = 0
        while (self.error > bounds[-1]) & (i <= 10):
            rank = round(rank * mult)
            Omega = xnp.randn(*(A.shape[0], rank), dtype=A.dtype, device=A.device)
            Lambda, self.U = get_nys_approx(A=A, Omega=Omega, eps=eps)
            self.error = estimate_approx_error(A, Lambda, self.U, tol=1e-7, max_iter=1000)
            i += 1
        if (self.error >= 0.) & (self.error < bounds[0]):
            rank = round(rank / mult)
        elif (self.error >= bounds[0]) & (self.error < bounds[1]):
            rank = rank
        elif (self.error >= bounds[1]) & (self.error < bounds[-1]):
            rank = round(rank * mult)
        self.rank = rank
        self._add_placeholders(Lambda, mu=mu, adjust_mu=adjust_mu)

    def _add_placeholders(self, Lambda, mu, adjust_mu):
        xnp = self.xnp
        self.adjusted_mu = amu = mu * xnp.max(Lambda) if adjust_mu else mu
        self.subspace_scaling = (xnp.min(Lambda) + amu) / (Lambda + amu) - 1
        self.subspace_scaling = self.subspace_scaling[:, None]
        self.preconditioned_eigmax = xnp.min(Lambda) + amu
        self.preconditioned_eigmin = amu

    def _matmat(self, V):
        subspace_term = self.U @ (self.subspace_scaling * (self.U.T @ V))
        return subspace_term + V


def select_rank_adaptively(A, rank_init, rank_max, tol, mult=2):
    xnp = A.xnp

    def cond_fun(state):
        _, error, *_, rank = state
        is_valid_rank = rank <= rank_max
        is_still_large = error > tol
        return is_still_large & is_valid_rank

    def body_fun(state):
        i, *_, rank = state
        rank = round(mult * rank)
        Omega = xnp.randn(*(A.shape[0], rank), dtype=A.dtype, device=A.device)
        Lambda, U = get_nys_approx(A, Omega, eps=1e-8)
        # error = xnp.jit(error_fn, static_argnums=(0,))(A, Lambda, U)
        error = error_fn(A, Lambda, U)
        return (i + 1, error, Lambda, U, rank)

    def error_fn(A, Lambda, U):
        return estimate_approx_error(A, Lambda, U, tol=1e-7, max_iter=1000)

    Omega = xnp.randn(*(A.shape[0], rank_init), dtype=A.dtype, device=A.device)
    Lambda, U = get_nys_approx(A, Omega, eps=1e-8)
    error = error_fn(A, Lambda, U)
    init_val = (0, xnp.abs(error), Lambda, U, rank_init)
    *_, Lambda, U, rank = xnp.while_loop_no_jit(cond_fun, body_fun, init_val)
    return Lambda, U, rank


def estimate_approx_error(A, Lambda, U, tol, max_iter):
    xnp = A.xnp
    Diag = Lambda[:, None]

    def matmat(V):
        return A @ V - U @ (Diag * (U.T @ V))

    E = LinearOperator(dtype=A.dtype, shape=A.shape, matmat=matmat)
    _, error, _ = power_iteration(E, tol=tol, max_iter=max_iter)
    return xnp.abs(error)

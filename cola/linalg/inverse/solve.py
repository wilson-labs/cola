from cola.utils import export
from cola.ops import LinearOperator
from cola.linalg.inverse.inv import inv


@export
def solve(A: LinearOperator, rhs, **kwargs):
    """Solves A x = b. This function uses cola.inverse().

    Args:
        A (LinearOperator): The linear operator to compute the inverse of.
        rhs (array): The right-hand side of the linear system.
         rhs could be of size (N,) or (N, B), where B is the number of
         systems to solve.
        tol (float, optional): The tolerance criteria. Defaults to 1e-6.
        P (array, optional): Preconditioner matrix
        x0 (array, optional): The initial guess for the solution
        pbar (bool, optional): Whether to show a progress bar. Defaults to False.
        max_iters (int, optional): The maximum number of iterations. Defaults to 5000.
        method (str, optional): Method to use, defaults to 'auto',
         options are 'auto', 'dense', 'iterative', 'svrg'.

    Returns:
        Array: The solution to the linear system.

    Example:
        >>> A = MyLinearOperator()
        >>> b = xnp.array()
        >>> x = cola.solve(A, b)

    """
    return inv(A, **kwargs) @ rhs

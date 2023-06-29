from plum import dispatch
from cola.linear_algebra import lazify
from cola.operators import SelfAdjoint
from cola.operators import Diagonal
from cola.operators import Kronecker
# from cola.linalg.eigs import eig


@dispatch
def sqrt(A: SelfAdjoint) -> SelfAdjoint:
    xnp = A.ops
    # eig_vals, eig_vecs = eig(A)
    eig_vals, eig_vecs = xnp.eigh(A.to_dense())
    Lambda = Diagonal(xnp.sqrt(eig_vals))
    Q = lazify(eig_vecs)
    return SelfAdjoint(Q @ Lambda @ Q.T)


@dispatch
def sqrt(A: Diagonal) -> Diagonal:
    xnp = A.ops
    return Diagonal(xnp.sqrt(A.diag))


@dispatch
def sqrt(A: Kronecker) -> Kronecker:
    Ms = [sqrt(Mi) for Mi in A.Ms]
    return Kronecker(*Ms)

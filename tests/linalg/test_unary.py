import numpy as np
import scipy.linalg
import cola.linalg.unary.unary as colau
from cola.linalg.unary.unary import apply_unary
from operator_market import op_names, get_test_operator
from cola.annotations import SelfAdjoint
from cola.ops import LinearOperator
from cola.linalg.algorithm_base import Auto
from cola.linalg.decompositions.decompositions import Lanczos
from cola.linalg.decompositions.decompositions import Arnoldi
from cola.utils.test_utils import parametrize, relative_error
from cola.backends import all_backends


@parametrize(all_backends, ['float64'], op_names, ['exp', 'sqrt']).excluding['torch', :, 'psd_kron', :]
def test_unary(backend, precision, op_name, fn_name):
    operator = get_test_operator(backend, precision, op_name)
    fn = getattr(colau, fn_name)
    spfn = getattr(scipy.linalg, fn_name + 'm')
    A, dtype, xnp = operator, operator.dtype, operator.xnp
    # ###############################################
    # TODO: take this ugly fix
    fnX = A.xnp.exp if fn_name == "exp" else A.xnp.sqrt
    fn = lambda X, ALG: apply_unary(fnX, X, ALG)
    # ###############################################
    A2 = LinearOperator(A.dtype, A.shape, A._matmat)
    tol = 1e-4
    v = xnp.randn(A.shape[-1], dtype=dtype, device=None)
    Adense = A.to_dense()
    if fn_name == 'sqrt' and xnp.iscomplexobj(Adense):
        return
    if op_name == 'square_kronecker' and backend == 'numpy':
        return
    Anp = np.array(Adense)
    fv = spfn(Anp) @ np.array(v)
    fv1 = np.array(fn(A, Auto(tol=tol)) @ v)
    e1 = relative_error(fv, fv1)
    assert e1 < 3 * tol, f"Dispatch rules failed on {type(A)} with error {e1}"
    if A.isa(SelfAdjoint):
        A3 = SelfAdjoint(A2)
        alg = Lanczos(tol=tol)
    else:
        A3 = A2
        alg = Arnoldi(tol=tol)
    if np.prod(A.shape) < 1000:
        fv2 = np.array(fn(A3, Auto(tol=tol)) @ v)
        e2 = relative_error(fv, fv2)
        assert e2 < 3 * tol, f"Dense f(A) failed on {type(A)} with error {e2}"
    diag = xnp.diag(Adense)
    not_scalarmul = relative_error(xnp.diag(diag.mean() + 0. * diag), Adense) > 1e-5
    if not_scalarmul:
        fv3 = np.array(fn(A3, alg) @ v)
        e3 = relative_error(fv, fv3)
        assert e3 < 3 * tol, f"SLQ logdet failed on {type(A)} with error {e3}"

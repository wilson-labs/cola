import cola
from cola.utils_test import get_xnp, parametrize, relative_error
from cola.ops import Tridiagonal, Dense, Diagonal, Product, ScalarMul
from functools import partial

@parametrize(['jax']) # many torch fns not yet implemented for vmap
def test_vmappable_constructor(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    # Certain functions like tridiagonal need to be redesigned to accomodate vmap
    #a,b,c = xnp.randn(3,5,10,key = xnp.PRNGKey(0))
    #Ts = xnp.vmap(Tridiagonal)(a[:,:-1],b,c[:,:-1])
    def make_operator(X1,X2):
        return cola.kron(Dense(X1),Diagonal(X2))
    k0 = xnp.PRNGKey(2)
    Ts = xnp.vmap(make_operator)(xnp.randn(4,5,5,key=k0),xnp.randn(4,2,key=k0))
    X = xnp.randn(4,10,2,key = xnp.PRNGKey(1))
    MM = lambda T, X: T@X
    bmm = xnp.vmap(MM)(Ts,X)
    dT = xnp.vmap(lambda A: A.to_dense())(Ts)
    assert relative_error(dT@X,bmm) < 1e-6

    vecX = X[...,0]
    bmm2 = xnp.vmap(MM)(Ts,vecX)
    assert relative_error((dT@vecX[...,None])[...,0],bmm2) < 1e-6   


@parametrize(['jax','torch'])
def test_jittable_constructor(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    #a,b,c = xnp.randn(3,5,10,key = xnp.PRNGKey(0))
    #Ts = xnp.vmap(Tridiagonal)(a[:,:-1],b,c[:,:-1])
    def make_operator(X1,X2):
        return cola.kron(Dense(X1),Diagonal(X2))
    k0 = xnp.PRNGKey(2)
    Ts = xnp.jit(make_operator)(xnp.randn(5,5,key=k0),xnp.randn(2,key=k0))
    X = xnp.randn(10,2,key = xnp.PRNGKey(1))
    MM = lambda T, X: T@X
    bmm = xnp.jit(MM)(Ts,X)
    dT = xnp.jit(lambda A: A.to_dense())(Ts)
    assert relative_error(dT@X,bmm) < 1e-6


@parametrize(['jax'])
def test_vmapped_linalg(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    #a,b,c = xnp.randn(3,5,10,key = xnp.PRNGKey(0))
    #Ts = xnp.vmap(Tridiagonal)(a[:,:-1],b,c[:,:-1])
    def make_operator(X1,X2):
        return cola.kron(Dense(X1),Diagonal(X2))
    k0 = xnp.PRNGKey(2)
    Ts = xnp.vmap(make_operator)(xnp.randn(4,5,5,key=k0),xnp.randn(4,2,key=k0))
    logdets = xnp.vmap(partial(cola.linalg.logdet))(Ts)
    logdets2 = xnp.slogdet(xnp.vmap(lambda T: T.to_dense())(Ts))[1]
    assert relative_error(logdets,logdets2) < 1e-6


@parametrize(['jax','torch'])
def test_grad(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    e1 = xnp.eye(10,dtype=dtype)[1]
    def f(A):
        return A[1,1]
    
    D = Diagonal(xnp.arange(10,dtype=dtype))
    g = xnp.grad(f)(D)
    assert relative_error(g.diag, e1) < 1e-6

    g = xnp.grad(f)(Product(D,ScalarMul(3.,D.shape,D.dtype)))
    g1,g2 = xnp.tree_flatten(g)[0]
    assert relative_error(g1, 3*e1) < 1e-6
    assert relative_error(g2, xnp.ones((1,), dtype=dtype))<1e-6
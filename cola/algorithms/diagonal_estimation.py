import cola
from cola.utils import export
from cola.ops import I_like,LinearOperator
import numpy as np

def get_I_chunk_like(A: LinearOperator, i, bs, shift=0):
    xnp = A.ops
    k=  shift
    if k<=0:
        k = abs(k)
        I_chunk = I_like(A)[:,i:i+bs+k].to_dense()
        padded_chunk = A.ops.zeros((A.shape[0],bs+k),dtype=A.dtype)
        slc = np.s_[:I_chunk.shape[-1]]
        padded_chunk = xnp.update_array(padded_chunk, I_chunk, slice(0,None),slc)
        chunk = I_chunk[:,:bs]
        shifted_chunk = padded_chunk[:,k:k+bs]
    else:
        I_chunk = I_like(A)[:,max(i-k,0):i+bs].to_dense()
        padded_chunk = A.ops.zeros((A.shape[0],bs+k),dtype=A.dtype)
        slc = np.s_[-I_chunk.shape[-1]:]
        padded_chunk = xnp.update_array(padded_chunk, I_chunk, slice(0,None),slc)
        chunk = I_chunk[:,-bs:]
        shifted_chunk = padded_chunk[:,:bs]
    return chunk, shifted_chunk

@export
def exact_diag(A:LinearOperator,k=0,bs=100):
    """ Extract the (kth) diagonal of a linear operator. """
    bs = min(100,A.shape[0])
    # lazily create chunks of the identity matrix of size bs
    diag_sum = 0
    xnp = A.ops
    for i in range(0, A.shape[0], bs):
        chunk, shifted_chunk = get_I_chunk_like(A,i,bs,k)
        diag_sum += ((A @ chunk)*shifted_chunk).sum(-1)
    if k <= 0:
        return diag_sum[abs(k):]
    else:
        return diag_sum[:(-k or None)]

@export
def approx_diag(A:LinearOperator,k=0,bs=100,tol=3e-2,max_iter=10000,pbar=False,info=False):
    """ Extract the (kth) diagonal of a linear operator using stochastic estimation """
    bs = min(100,A.shape[0])
    # lazily create chunks of the identity matrix of size bs
    xnp = A.ops
    assert tol > 1e-3, "tolerance chosen too high for stochastic diagonal estimation"

    @xnp.jit
    def body(state):
        #TODO: fix randomness when using with jax
        i, diag_sum, diag_sumsq = state
        z = xnp.randn(A.shape[0],bs,dtype=A.dtype)
        z2 = xnp.roll(z,-k,0)
        z2 = xnp.update_array(z2, 0, slice(0,abs(k)) if k<=0 else slice(-abs(k),None))
        slc = slice(abs(k),None) if -k>0 else slice(None,-abs(k) or None)
        estimator = ((A @ z)*z2)[slc]
        #print(diag_sum/((i+1)*bs))
        return i + 1, diag_sum+estimator.sum(-1), diag_sumsq+(estimator**2).sum(-1)

    def err(state):
        i, diag_sum, diag_sumsq = state
        mean = diag_sum/(i*bs)
        stderr = xnp.sqrt((diag_sumsq/(i*bs) - mean**2)/(i*bs))
        return xnp.mean(stderr/xnp.maximum(xnp.abs(mean),.1*xnp.ones_like(mean)))
    
    def cond(state):
        return not state[0] or (state[0] < max_iter) & (err(state) > tol)
    
    while_loop, infos = xnp.while_loop_winfo(err, tol, pbar=pbar)
    zeros = xnp.zeros((A.shape[0]-abs(k),),dtype=A.dtype)
    i,diag_sum,_ =state= while_loop(cond, body, (0, zeros, zeros))
    mean = diag_sum/(i*bs)
    return mean

        


#     while_loop, infos = xnp.while_loop_winfo(lambda state: state[0] < A.shape[0], tol, pbar=False)


    


# %%

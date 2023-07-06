
#%%
import cola
from cola.utils import export
from cola.ops import I_like,LinearOperator
import numpy as np

def exact_diag(A:LinearOperator,k=0,bs=100):
    """ Extract the (kth) diagonal of a linear operator. """
    bs = min(100,A.shape[0])
    # lazily create chunks of the identity matrix of size bs
    diag_sum = 0
    xnp = A.ops
    for i in range(0, A.shape[0], bs):
        if k < 0:
            kabs = abs(k)
            I_chunk = I_like(A)[:,i:i+bs+kabs].to_dense()
            padded_chunk = A.ops.zeros((A.shape[0],bs+kabs),dtype=A.dtype)
            slc = np.s_[:I_chunk.shape[-1]]
            padded_chunk = xnp.update_array(padded_chunk, I_chunk, slice(0,None),slc)
            chunk = I_chunk[:,:bs]
            shifted_chunk = padded_chunk[:,kabs:kabs+bs]
            diag_sum += ((A @ chunk)*shifted_chunk).sum(-1)[kabs:]
        else:
            I_chunk = I_like(A)[:,max(i-k,0):i+bs].to_dense()
            padded_chunk = A.ops.zeros((A.shape[0],bs+k),dtype=A.dtype)
            slc = np.s_[-I_chunk.shape[-1]:]
            padded_chunk = xnp.update_array(padded_chunk, I_chunk, slice(0,None),slc)
            chunk = I_chunk[:,-bs:]
            shifted_chunk = padded_chunk[:,:bs]
            diag_sum += ((A @ chunk)*shifted_chunk).sum(-1)[:(-k or None)]

    return diag_sum

# create a little test
import cola
import numpy as np
from cola.ops import Diagonal,Identity,Sum,BlockDiag,ScalarMul,Dense
A = Dense(np.array([[1,2,3],[4,5,6],[7,8,9]]))
print(A.to_dense())
for u in [-2,-1,0,1,2]:
    print(exact_diag(A,k=u))
    print(np.diag(A.to_dense(),k=u))

# %%

#%%
from cola.annotations import SelfAdjoint
from cola.ops import Dense
import numpy as np
A = Dense(np.random.randn(10, 10))
print(A.annotations)
# %%
B = SelfAdjoint(A)
print(B.annotations)
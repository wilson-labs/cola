
# #%%
# import cola
# from cola.utils import export
# from cola.ops import I_like,LinearOperator

# # def exact_diag(A:LinearOperator,k=0,bs=100):
# #     """ Extract the (kth) diagonal of a linear operator. """
# #     bs = 100
# #     # lazily create chunks of the identity matrix of size bs
# #     diag_sum = 0
# #     for i in range(0, A.shape[0], bs):
# #         I_chunk = I_like(A)[:,i:i+bs+k].to_dense()
# #         padded_chunk = A.ops.zeros((A.shape[0],bs+k))
# #         padded_chunk[:,:I_chunk.shape[-1]] = I_chunk
# #         chunk = I_chunk[:,:bs]
# #         shifted_chunk = padded_chunk[:,k:k+bs]
# #         diag_sum += ((A @ chunk)*shifted_chunk).sum(-1)
# #     return diag_sum
# # %%
# if __name__ == '__main__':
#     import cola
#     import numpy as np
#     from cola.ops import Diagonal,Identity,Sum,BlockDiag,ScalarMul,Dense
#     A = Dense(np.array([[1,2,3],[4,5,6],[7,8,9]]))
#     print(exact_diag(A))
# # create a little test
# # import cola
# # import numpy as np
# # from cola.ops import Diagonal,Identity,Sum,BlockDiag,ScalarMul,Dense
# # A = Dense(np.array([[1,2,3],[4,5,6],[7,8,9]]))
# # #print(exact_diag(A))
# # print(np.diag(A))


# %%

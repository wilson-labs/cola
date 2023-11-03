import numpy as np
from cola.utils.test_utils import generate_spectrum, generate_pd_from_diag
from cola.utils.test_utils import relative_error


def test_shifted_qr():
    np.set_printoptions(formatter={"float": "{:0.2f}".format})
    np_dtype = np.float64
    N = 10
    diag = generate_spectrum(coeff=0.5, scale=1.0, size=N, dtype=np_dtype)
    A = generate_pd_from_diag(diag, dtype=diag.dtype, seed=21)

    A_new = shifted_qr_np(A, shifts=np.zeros((50, )))
    rel_error = relative_error(np.diag(A_new), diag)
    assert rel_error < 1e-8


def shifted_qr_np(A, shifts):
    N = A.shape[0]
    for j in range(len(shifts)):
        Q, R = np.linalg.qr(A - shifts[j] * np.eye(N), mode="complete")
        A = R @ Q + shifts[j] * np.eye(N)
    return A

import numpy as np
from cola.utils.test_utils import generate_spectrum, generate_pd_from_diag
from cola.utils.test_utils import relative_error


def test_shifted_qr_np():
    np.set_printoptions(formatter={"float": "{:0.2f}".format})
    np_dtype = np.float64
    N = 10
    diag = generate_spectrum(coeff=0.5, scale=1.0, size=N, dtype=np_dtype)
    A = generate_pd_from_diag(diag, dtype=diag.dtype, seed=21)

    A_new, _ = shifted_qr_np(A, shifts=np.zeros((50, )))
    approx = np.sort(np.diag(A_new))[::-1]
    rel_error = relative_error(approx, diag)
    assert rel_error < 1e-8

    A_new, Q = shifted_qr_np(A, shifts=diag[5:])
    approx = Q @ A_new @ Q.T
    rel_error = relative_error(approx, A)
    assert rel_error < 1e-12

    A_new, _ = shifted_qr_np(A_new, shifts=np.zeros((40, )))
    approx = np.sort(np.diag(A_new))[::-1][:5]
    rel_error = relative_error(approx, diag[:5])
    assert rel_error < 1e-8

    A_new, _ = shifted_qr_np(A, shifts=diag[2] * np.ones((6, )))
    approx = np.sort(np.diag(A_new))[::-1][2]
    rel_error = relative_error(approx, diag[2])
    print(f"Rel error: {rel_error:2.5e}")
    assert rel_error < 1e-8


def shifted_qr_np(A, shifts):
    N = A.shape[0]
    V, Id = np.eye(N), np.eye(N)

    for j in range(len(shifts)):
        Q, R = np.linalg.qr(A - shifts[j] * Id, mode="complete")
        A = R @ Q + shifts[j] * Id
        V @= Q
    return A, V

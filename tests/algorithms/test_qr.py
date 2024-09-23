import numpy as np

from cola.backends import tracing_backends
from cola.linalg.tbd.qr import shifted_qr
from cola.ops import Dense
from cola.utils.utils_for_tests import generate_pd_from_diag, generate_spectrum, get_xnp, parametrize, relative_error


@parametrize(tracing_backends)
def test_shifted_qr(backend):
    xnp = get_xnp(backend)
    dtype, np_dtype = xnp.float32, np.float32
    N = 5
    diag_np = generate_spectrum(coeff=0.5, scale=1.0, size=N, dtype=np_dtype)
    A_np = generate_pd_from_diag(diag_np, dtype=diag_np.dtype, seed=21)
    A = Dense(xnp.array(A_np, dtype=dtype, device=None))

    shifts = xnp.zeros((50, ), dtype=dtype, device=None)
    A_new, _ = shifted_qr(A, shifts=shifts)
    approx = np.array(xnp.diag(A_new.to_dense()), dtype=np_dtype)
    rel_error = relative_error(approx, diag_np)
    print(f"\nRel error: {rel_error:2.5e}")
    assert rel_error < 1e-6

    shifts = xnp.array(diag_np[:5], dtype=dtype, device=None)
    H, V = shifted_qr(A, shifts=shifts)
    rel_error = relative_error(A.to_dense(), (V @ H @ V.T).to_dense())
    print(f"Rel error: {rel_error:2.5e}")
    assert rel_error < 1e-6

    for idx in range(1, 6):
        shifts = xnp.array(diag_np[:idx], dtype=dtype, device=None)
        H, V = shifted_qr(A, shifts=shifts)
        H_np, V_np = shifted_qr_np(A_np, shifts=np.array(shifts, dtype=np_dtype))
        rel_error = relative_error(H_np, np.array(H.to_dense()))
        print(f"Rel error: {rel_error:2.5e}")
        assert rel_error < 1e+1
        rel_error = relative_error(V_np, np.array(V.to_dense()))
        print(f"Rel error: {rel_error:2.5e}")
        assert rel_error < 1e+1


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
    V, Id = np.eye(N, dtype=A.dtype), np.eye(N, dtype=A.dtype)

    for j in range(len(shifts)):
        Q, R = np.linalg.qr(A - shifts[j] * Id, mode="complete")
        A = R @ Q + shifts[j] * Id
        V @= Q
    return A, V

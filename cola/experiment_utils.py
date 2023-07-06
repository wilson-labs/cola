from math import floor
import time
from pathlib import Path
import os
import pickle
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import eigs as scipy_eigs
from scipy.sparse.linalg import LinearOperator as LOS
from jax.scipy.sparse.linalg import gmres as jgmres
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import load_iris
from sklearn.linear_model import Ridge
from functools import partial
import numpy as np
import pandas as pd
from emlp.reps import Rep
from emlp.groups import S
from emlp.nn.objax import uniform_rep
from emlp.reps.representation import orthogonal_complement
from emlp.reps.representation import krylov_constraint_solve
from jax.config import config
from jax import numpy as jnp
from jax import jit, jvp, vmap, jacfwd
from jax.scipy.signal import correlate
from scipy.io import loadmat
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from cola.ops import SelfAdjoint
from cola.ops import Diagonal
from cola.ops import ScalarMul
from cola.ops import I_like
from cola.ops import CustomLinOp
from cola.ops import LinearOperator
from cola.linalg.eigs import eig
from cola.basic_operations import lazify
from cola.basic_operations import kron
from cola.linalg.inverse import inverse
from cola.algorithms.svrg import solve_svrg_rff
from cola.algorithms.svd import randomized_svd
from cola.algorithms.cg import run_batched_tracking_cg
from cola.utils_test import generate_spectrum
from cola.utils_test import generate_clustered_spectrum
from cola.utils_test import generate_diagonals
from cola.utils_test import generate_pd_from_diag


def get_times_pca_sk(X, pca_num, results, repeat):
    times = np.zeros(shape=(repeat, ))
    for idx in range(repeat):
        t0 = time.time()
        _ = PCA(n_components=pca_num, svd_solver="randomized").fit(X)
        t1 = time.time()
        times[idx] = t1 - t0

    results["sklearn"] = {"times": times, "system_size": X.shape[1]}
    print("*+" * 50)
    print(f"Times   {np.mean(times[1:]):1.5e} sec")
    print("*+" * 50)


def get_times_pca(XTX, args, results, repeat):
    pca_num, rank = args
    times = np.zeros(shape=(repeat, ))
    for idx in range(repeat):
        t0 = time.time()
        eigs, *_ = randomized_svd(XTX, rank)
        eigs[:pca_num]
        t1 = time.time()
        times[idx] = t1 - t0

    results["cola"] = {"times": times, "system_size": XTX.shape[0], "eigs": eigs}
    print("*+" * 50)
    print(f"Times   {np.mean(times[1:]):1.5e} sec")
    print("*+" * 50)


def get_times_sk_linear(X, y, mu, results, repeat):
    times = np.zeros(shape=(repeat, ))
    res = np.zeros(shape=(repeat, 2))
    for idx in range(repeat):
        t0 = time.time()
        reg = Ridge(alpha=mu, solver="lsqr").fit(X, y)
        t1 = time.time()
        times[idx] = t1 - t0
        res[idx, 0] = np.linalg.norm(X @ reg.coef_ - y)
        res[idx, 1] = res[idx, 0] / np.linalg.norm(y)

    results["sklearn"] = {"times": times, "system_size": X.shape[1], "soln": reg.coef_}
    print("*+" * 50)
    print(f"Times   {np.mean(times[1:]):1.5e} sec")
    print(f"Res Rel {np.mean(res[:, 1]):1.5e}")
    print("*+" * 50)


def get_times_schrodinger(H, args, results, repeat):
    k, ndims = args
    N, N2 = H.shape[0], H.shape[0]**ndims
    times = np.zeros(shape=(repeat, ))
    for idx in range(repeat):
        t0 = time.time()
        e2, _, _ = eig(H, method='arnoldi', max_iters=int(N * 1.))
        print("Arnoldi", np.sort(np.abs(e2))[:k])
        t1 = time.time()
        times[idx] = t1 - t0

    results["cola"] = {"times": times, "grid_size": N2, "points": N}
    print("*+" * 50)
    print(f"cola | grid size {N2:,d}")
    print(f"Times   {np.mean(times[1:]):1.5e} sec")
    print("*+" * 50)


def get_times_schrodinger_jax(H, args, results, repeat):
    k, ndims = args
    N, N2 = H.shape[0], H.shape[0]**ndims
    times = np.zeros(shape=(repeat, ))
    for idx in range(repeat):
        t0 = time.time()
        e2, *_ = scipy_eigs(H, k=int(N) - 2)
        print("SciPy Arnoldi", np.sort(np.abs(e2))[:k])
        t1 = time.time()
        times[idx] = t1 - t0

    results["scipy"] = {"times": times, "grid_size": N2, "points": N}
    print("*+" * 50)
    print(f"SciPy | grid size {N2:,d}")
    print(f"Times   {np.mean(times[1:]):1.5e} sec")
    print("*+" * 50)


def get_times_minimal_jax(J_matvec, pde_op, args, results, repeat, tol=1e-3):
    domain, x = args
    N, N2 = x.shape[0], x.shape[0] * x.shape[1]
    times = np.zeros(shape=(repeat, ))
    for idx in range(repeat):
        t0 = time.time()
        err = np.inf
        z = jnp.zeros_like(x[domain]).reshape(-1)
        while err > tol:
            Jmvm = partial(J_matvec, z)
            F = pde_op(z)
            err = jnp.max(jnp.abs(F))
            delta, info = jgmres(Jmvm, -F, tol=tol)
            z += delta
            text = f"PDE Error: {err:1.1e}, "
            text += f"Update size: {jnp.linalg.norm(delta):1.1e}, info: {info}"
            print(text)
        t1 = time.time()
        times[idx] = t1 - t0

    results["jax"] = {"times": times, "grid_size": N2, "points": N, "z": z}
    print("*+" * 50)
    print(f"SciPy JAX | grid size {N2:,d}")
    print(f"Times   {np.mean(times[1:]):1.5e} sec")
    print("*+" * 50)


def get_times_minimal_scipy(J_matvec, pde_op, args, results, repeat, tol=1e-3):
    domain, x = args
    N, N2 = x.shape[0], x.shape[0] * x.shape[1]
    times = np.zeros(shape=(repeat, ))
    for idx in range(repeat):
        t0 = time.time()
        err = np.inf
        z = jnp.zeros_like(x[domain]).reshape(-1)
        while err > tol:
            Jmvm = partial(J_matvec, z)
            F = pde_op(z)
            err = jnp.max(jnp.abs(F))
            shape = (int(domain.sum()), int(domain.sum()))

            J = LOS(shape, matvec=Jmvm, matmat=jit(vmap(Jmvm, -1, -1)), dtype=np.float32)
            delta, info = gmres(J, -F, tol=tol)
            z += delta
            text = f"PDE Error: {err:1.1e}, "
            text += f"Update size: {jnp.linalg.norm(delta):1.1e}, info: {info}"
            print(text)
        t1 = time.time()
        times[idx] = t1 - t0

    results["scipy"] = {"times": times, "grid_size": N2, "points": N, "z": z}
    print("*+" * 50)
    print(f"SciPy| grid size {N2:,d}")
    print(f"Times   {np.mean(times[1:]):1.5e} sec")
    print("*+" * 50)


def get_times_minimal_surface(J_matvec, pde_op, args, results, repeat, tol=1e-3):
    domain, x = args
    N, N2 = x.shape[0], x.shape[0] * x.shape[1]
    times = np.zeros(shape=(repeat, ))
    for idx in range(repeat):
        t0 = time.time()
        err = np.inf
        z = jnp.zeros_like(x[domain]).reshape(-1)
        while err > tol:
            Jmvm = partial(J_matvec, z)
            F = pde_op(z)
            err = jnp.max(jnp.abs(F))
            shape = (int(domain.sum()), int(domain.sum()))
            J = CustomLinOp(dtype=jnp.float32, shape=shape, matmat=jit(vmap(Jmvm, -1, -1)))
            J_inv = inverse(J, tol=tol, max_iters=F.shape[0] // 2, method="gmres", info=True)
            delta = J_inv @ -F
            info = J_inv.info
            z += delta
            text = f"PDE Error: {err:1.1e}, "
            text += f"Update size: {jnp.linalg.norm(delta):1.1e}, info: {info}"
            print(text)
        t1 = time.time()
        times[idx] = t1 - t0

    results["cola"] = {"times": times, "grid_size": N2, "points": N, "z": z}
    print("*+" * 50)
    print(f"cola | grid size {N2:,d}")
    print(f"Times   {np.mean(times[1:]):1.5e} sec")
    print("*+" * 50)


def get_times_spectral_sklearn(sparse_data, results, repeat, eigen_solver="amg"):
    times = np.zeros(shape=(repeat, ))
    for idx in range(repeat):
        t0 = time.time()
        SC = SpectralClustering(affinity="precomputed", eigen_solver=eigen_solver, n_init=1)
        SC.fit(sparse_data)
        t1 = time.time()
        times[idx] = t1 - t0
    results[eigen_solver] = {
        "times": times,
        "edges": sparse_data.data.shape[0],
        "nodes": sparse_data.shape[0]
    }

    print("*+" * 50)
    print(f"sklearn w {eigen_solver}")
    print(f"Times   {np.mean(times[1:]):1.5e} sec")
    print("*+" * 50)
    return results


def get_times_spectral(L, results, repeat, args):
    lanczos_iters, embedding_size, n_clusters, num_edges = args
    times = np.zeros(shape=(repeat, ))
    for idx in range(repeat):
        t0 = time.time()
        eigvals, eigvecs = eig(L, method="lanczos", max_iters=lanczos_iters)
        x_emb = eigvecs[:, :-embedding_size]
        kmeans = KMeans(n_clusters=n_clusters).fit(x_emb)
        t1 = time.time()
        times[idx] = t1 - t0
    results["cola"] = {
        "times": times,
        "nodes": L.shape[0],
        "edges": num_edges,
        "eigvals": eigvals,
        "x_emb": x_emb,
        "labels": kmeans.labels_
    }
    print("*+" * 50)
    print("cola")
    print(f"Times   {np.mean(times[1:]):1.5e} sec")
    print("*+" * 50)
    return results


def get_times_emlp(A, results, repeat, key):
    times = np.zeros(shape=(repeat, ))
    for idx in range(repeat):
        t0 = time.time()
        soln = A.equivariant_basis()
        rhs = jnp.ones(shape=(soln.shape[1]))
        out = soln @ rhs
        out.block_until_ready()
        print(soln.shape)
        t1 = time.time()
        times[idx] = t1 - t0
    results[key] = {"times": times, "system_size": soln.shape[0]}
    print("*+" * 50)
    print(key)
    skip = 1 if repeat > 1 else 0
    print(f"Times   {np.mean(times[skip:]):1.5e} sec")
    print("*+" * 50)


def get_times_svrg(A, rhs, solver_kwargs, results, xnp, repeat):
    times = np.zeros(shape=(repeat, ))
    res = np.zeros(shape=(repeat, 2))
    for idx in range(repeat):
        t0 = time.time()
        # soln, info = svrg_solveh(A, rhs, **solver_kwargs)
        soln, info = solve_svrg_rff(A, rhs, **solver_kwargs)
        soln.block_until_ready()
        t1 = time.time()
        times[idx] = t1 - t0
        res[idx, 0] = xnp.norm(A @ soln - rhs)
        res[idx, 1] = res[idx, 0] / xnp.norm(rhs)
    results["svrg"] = {
        "res": res,
        "times": times,
        "system_size": A.shape[0],
        "errors": info["errors"]
    }
    print("*+" * 50)
    print("SVRG")
    print(f"Times   {np.mean(times[1:]):1.5e} sec")
    print(f"Res Rel {np.mean(res[:, 1]):1.5e}")
    print("*+" * 50)
    return results


def get_times_cg(A, rhs, solver_kwargs, results, xnp, repeat):
    times = np.zeros(shape=(repeat, ))
    res = np.zeros(shape=(repeat, 2))
    Kinv = inverse(A, **solver_kwargs)
    for idx in range(repeat):
        t0 = time.time()
        soln = Kinv @ rhs
        if xnp.__name__.find("jax") >= 0:
            soln.block_until_ready()
        t1 = time.time()
        times[idx] = t1 - t0
        res[idx, 0] = xnp.norm(A @ soln - rhs)
        res[idx, 1] = res[idx, 0] / xnp.norm(rhs)
    results["iterative"] = {
        "res": res,
        "times": times,
        "system_size": A.shape[0],
        "errors": Kinv.info["residuals"]
    }
    print("*+" * 50)
    print("Iterative")
    print(f"Times   {np.mean(times[1:]):1.5e} sec")
    print(f"Res Rel {np.mean(res[:, 1]):1.5e}")
    print("*+" * 50)
    return results


def get_times_cg2(A, rhs, solver_kwargs, results, xnp, repeat):
    times = np.zeros(shape=(repeat, ))
    res = np.zeros(shape=(repeat, 2))
    P = I_like(A)
    x0 = xnp.zeros_like(rhs)
    for idx in range(repeat):
        t0 = time.time()
        # soln, info = cg(A, rhs, **solver_kwargs)
        soln, _, _, info = run_batched_tracking_cg(A, rhs, x0=x0, preconditioner=P, **solver_kwargs)
        errors = info[0][:, 0, 0]
        mask = errors > 0.
        errors = errors[mask]
        if xnp.__name__.find("jax") >= 0:
            soln.block_until_ready()
        t1 = time.time()
        times[idx] = t1 - t0
        res[idx, 0] = xnp.norm(A @ soln - rhs)
        res[idx, 1] = res[idx, 0] / xnp.norm(rhs)
    results["iterative"] = {
        "res": res,
        "times": times,
        "system_size": A.shape[0],
        # "errors": Kinv.info["residuals"],
        # "errors": info["residuals"],
        "errors": errors,
    }
    print("*+" * 50)
    print("Iterative")
    print(f"Times   {np.mean(times[1:]):1.5e} sec")
    print(f"Res Rel {np.mean(res[:, 1]):1.5e}")
    print("*+" * 50)
    return results


def get_times(A, rhs, solver_kwargs, results, xnp, repeat, key):
    times = np.zeros(shape=(repeat, ))
    res = np.zeros(shape=(repeat, 2))
    Kinv = inverse(A, **solver_kwargs)
    for idx in range(repeat):
        t0 = time.time()
        soln = Kinv @ rhs
        if xnp.__name__.find("jax") >= 0:
            soln.block_until_ready()
        t1 = time.time()
        times[idx] = t1 - t0
        res[idx, 0] = xnp.norm(A @ soln - rhs)
        res[idx, 1] = res[idx, 0] / xnp.norm(rhs)
    results[key] = {"res": res, "times": times, "system_size": A.shape[0], "soln": soln}
    print("*+" * 50)
    print(key)
    print(f"Times   {np.mean(times[1:]):1.5e} sec")
    print(f"Res Rel {np.mean(res[:, 1]):1.5e}")
    print("*+" * 50)
    return results


def get_dense_times(A, rhs, results, N, xnp, repeat):
    times = np.zeros(shape=(repeat, ))
    res = np.zeros(shape=(repeat, 2))
    A_dense = A.to_dense()
    for idx in range(repeat):
        t0 = time.time()
        soln = xnp.solve(A_dense, rhs)
        if xnp.__name__.find("jax") >= 0:
            soln.block_until_ready()
        t1 = time.time()
        times[idx] = t1 - t0
        res[idx, 0] = xnp.norm(A_dense @ soln - rhs)
        res[idx, 1] = res[idx, 0] / xnp.norm(rhs)
    results[N]["dense"] = {"res": res, "times": times, "system_size": A.shape[0]}
    print("*+" * 50)
    print("Dense")
    print(f"Times {np.mean(times[1:]):1.5e} sec")
    print(f"Res   {np.mean(res):1.5e}")
    print("*+" * 50)
    del A_dense
    return results


def construct_schrodinger_setting(N, ndims):
    def square_compactification(x):
        return jnp.arctan(x) * 2 / jnp.pi

    def inv_square_compactification(y):
        return jnp.tan(y * jnp.pi / 2)

    grid = jnp.linspace(-1 + .001, 1 - .001, N)
    wgrid = jnp.stack(jnp.meshgrid(*(ndims * [grid])), axis=-1).reshape(-1, ndims)
    T = square_compactification
    Tinv = inv_square_compactification
    xyz = vmap(Tinv)(wgrid)
    print(xyz[0], xyz[-1])
    DT = vmap(jacfwd(T))(xyz)
    laplacian_factor2 = DT @ DT.transpose((0, 2, 1))
    laplacian_factor1 = vmap(lambda z: (jacfwd(jacfwd(T))(z) * jnp.eye(ndims)[None, :, :]).sum(
        (1, 2)))(xyz)
    dw = grid[1] - grid[0]
    deriv = jnp.array([-1 / 2, 0., 1 / 2]) / dw

    def hdiag(x):
        def cderiv(x):
            return correlate(x, jnp.array([1., -2, 1.]) / dw**2, mode='same')

        dds = jnp.stack([jnp.apply_along_axis(cderiv, i, x).reshape(-1) for i in range(ndims)],
                        axis=0)
        embedded_diag = vmap(jnp.diag, -1, -1)(dds).transpose((2, 0, 1))
        return embedded_diag

    def jderiv(x):
        return correlate(x, deriv, mode='same')

    def di(x, i):
        return jnp.apply_along_axis(jderiv, i, x)

    def d(x, axis=-1):
        return jnp.stack([di(x, i) for i in range(ndims)], axis=axis)

    def vfn(x):
        return (x * x).sum() / 2

    @jit
    def laplacian(psi):
        psi_grid = psi.reshape(*(ndims * (N, )))
        dpsi = d(psi_grid)
        hessian = d(dpsi).reshape(-1, ndims, ndims)
        hessian = jnp.where(jnp.eye(ndims)[None] + 0 * hessian > 0.5, hdiag(psi_grid), hessian)
        l1 = (dpsi.reshape(-1, ndims) * laplacian_factor1).sum(-1)
        l2 = (hessian * laplacian_factor2).sum((1, 2))
        return (l1 + l2).reshape(psi.shape)

    L = LinearOperator(jnp.float64, shape=(N**ndims, N**ndims), matmat=jit(vmap(laplacian, -1, -1)))
    v = vmap(vfn)(xyz).reshape(-1)
    V = Diagonal(v)
    H = -L / 2 + V
    return H


def construct_minimal_surface_setting(dx, boundary_vals, domain):
    @jit
    def pde_deriv(z):
        def deriv(x):
            return correlate(x, jnp.array([-1 / 2, 0, 1 / 2]) / dx, mode='same')

        def deriv2(x):
            return correlate(x, jnp.array([1., -2, 1.]) / dx**2, mode='same')

        zx, zy = [jnp.apply_along_axis(deriv, i, z) for i in [0, 1]]
        zxx, zyy = [jnp.apply_along_axis(deriv2, i, z) for i in [0, 1]]
        zxy = jnp.apply_along_axis(deriv, 1, zx)
        return (1 + zx**2) * zyy - 2 * zx * zy * zxy + (1 + zy**2) * zxx

    @jit
    def pde_op(u):
        padded_domain = jnp.zeros(boundary_vals.shape) + boundary_vals
        padded_domain = padded_domain.at[domain].set(u.reshape(-1))
        padded_domain = pde_deriv(padded_domain)
        return padded_domain[domain].reshape(u.shape)

    @jit
    def J_matvec(u, v):
        return jvp(pde_op, (u, ), (v, ))[1]

    return J_matvec, pde_op


def load_uci_data(data_dir, dataset, train_p=0.75, test_p=0.15):
    file_path = os.path.join(data_dir, dataset + '.mat')
    data = np.array(loadmat(file_path)['data'])
    X = data[:, :-1]
    y = data[:, -1]

    # good_dimensions = X.var(dim=-2) > 1.0e-10
    # if int(good_dimensions.sum()) < X.shape[1]:
    #     no_var_dim = X.size(1) - int(good_dimensions.sum())
    #     logging.info(f"Removed {no_var_dim:d} dimensions with no variance")
    #     X = X[:, good_dimensions]

    if dataset in ['keggundirected', 'slice']:
        X = SimpleImputer(missing_values=np.nan).fit_transform(X.data.numpy())
        X = np.array(X)

    X = X - X.min(0)[0]
    X = 2.0 * (X / X.max(0)[0]) - 1.0
    y -= y.mean()
    y /= y.std()

    train_n = int(floor(train_p * X.shape[0]))
    valid_n = int(floor((1. - train_p - test_p) * X.shape[0]))

    split = split_dataset(X, y, train_n, valid_n)
    train_x, train_y, valid_x, valid_y, test_x, test_y = split

    return train_x, train_y, test_x, test_y, valid_x, valid_y


def split_dataset(x, y, train_n, valid_n):
    train_x = x[:train_n, :]
    train_y = y[:train_n]

    valid_x = x[train_n:train_n + valid_n, :]
    valid_y = y[train_n:train_n + valid_n]

    test_x = x[train_n + valid_n:, :]
    test_y = y[train_n + valid_n:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y


def construct_pca_dataset1(xnp, dtype):
    X = load_iris()
    X = xnp.array(X.data, dtype=dtype)
    XTX = lazify(X.T / X.shape[0]) @ lazify(X)
    print(f"N={X.shape[0]} | D={X.shape[1]:,d}")
    return XTX


def construct_pca_dataset2(xnp, dtype):
    def PCA_dataset(n=10**5, rank=100, float_type=np.float64, gap=1e-4):
        del gap
        toy_eigs2 = np.random.rand(rank)**2
        toy_eigs2 = np.sort(toy_eigs2)[::-1]
        toy_eigs2[:30] += np.linspace(0, 1, 30)[::-1]
        toy_eigs2 += 1e-3 * np.max(toy_eigs2)
        # toy_eigs2[0] = toy_eigs2[1]+gap*toy_eigs2[1]
        print(toy_eigs2[:30])
        S = np.diag(toy_eigs2).astype(float_type)
        U = np.random.randn(n, rank).astype(float_type)
        U, _ = jnp.linalg.qr(U)
        V = np.random.randn(rank, rank).astype(float_type)
        V, _ = jnp.linalg.qr(V)
        X = (np.sqrt(n) * U @ S @ V.T).astype(float_type)
        A = X.T @ X / n
        return X, V, A, S

    n = 10**5
    rank = 500
    float_type = np.float64
    gap = 1e-2
    X, *_ = PCA_dataset(n, rank, float_type, gap)
    X = xnp.array(X, dtype=dtype)
    XTX = lazify(X.T / n) @ lazify(X)
    return XTX


def load_graph_data(filepath, num_edges=-1):
    df = pd.read_csv(filepath, skiprows=4, delimiter="\t", header=None, names=["to", "from"])
    df = df[:num_edges]
    df2 = pd.read_csv(filepath, skiprows=4, delimiter="\t", header=None, names=["from", "to"])
    df2 = df2[:num_edges]
    df_undir = pd.concat((df, df2), axis=0)
    df_undir = df_undir.drop_duplicates()
    id_map = map_nodes_to_id(df_undir["from"].unique())
    N = len(id_map)
    print(f"Found {N:,d} nodes")
    for col in ["from", "to"]:
        df_undir[col] = df_undir[col].map(id_map)
    data = np.ones(shape=len(df_undir))
    row, col = np.array(df_undir["to"]), np.array(df_undir["from"])
    sparse_matrix = csr_matrix((data, (row, col)), shape=(N, N))

    return sparse_matrix


def transform_to_csr(sparse_matrix, xnp, dtype):
    data = xnp.array(sparse_matrix.data, dtype=dtype)
    indices = xnp.array(sparse_matrix.indices, dtype=xnp.int64)
    indptr = xnp.array(sparse_matrix.indptr, dtype=xnp.int64)
    return data, indices, indptr, sparse_matrix.shape


def map_nodes_to_id(nodes):
    out = {}
    for idx in range(len(nodes)):
        out[int(nodes[idx])] = idx
    return out


def construct_emlp_ops(N):
    rep = uniform_rep(N, S(5))

    def wrap_rep(R, dense):
        class CustomRep(Rep):
            def __init__(self, dense=False):
                G = list(R.reps.keys())[0].G
                self.G = G
                self.dense = dense

            def size(self):
                return R.size()

            def rho(self, g):
                return R.rho(g)

            def __str__(self):
                return str(R)

            def equivariant_basis(self):
                if self == ScalarMul:
                    return jnp.ones((1, 1))
                canon_rep, perm = self.canonicalize()
                invperm = np.argsort(perm)

                C_lazy = canon_rep.constraint_matrix()
                if self.dense:
                    C_dense = C_lazy.to_dense()
                    result = orthogonal_complement(C_dense)
                else:
                    result = krylov_constraint_solve(C_lazy)
                return result[invperm]

        return CustomRep(dense)

    Rdense = wrap_rep(rep, dense=True)
    Riterative = wrap_rep(rep, dense=False)
    return rep, Rdense, Riterative


def generate_biposson_data(N, ndims):
    xgrid = jnp.linspace(-1, 1, N)[1:-1]
    N = len(xgrid)
    xygrid = jnp.stack(jnp.meshgrid(*(ndims * [xgrid])), axis=-1)
    dx = xgrid[1] - xgrid[0]
    x, y = xygrid.transpose((2, 0, 1))
    rho = ((1 - x**2) * (1 - y**2)) * ((x + y) * jnp.cos(4 * x) - 2 * x * y * jnp.sin(4 * x))
    rho = rho.reshape(-1)
    return rho, dx, N


def construct_laplacian(x, N, dx, ndims):
    z = x.reshape(ndims * (N, ))

    def cderiv(a):
        return correlate(a, jnp.array([1., -2, 1.]) / dx**2, mode='same')

    return -sum([jnp.apply_along_axis(cderiv, i, z) for i in range(ndims)]).reshape(-1)


def convert_results_to_df(results, var_name="system_size", skip=1, time_name="times"):
    df = []
    for key, value in results.items():
        for kk, vv in value.items():
            times = np.mean(vv[time_name][skip:])
            out = vv[var_name]
            if isinstance(out, np.ndarray):
                out = np.mean(out[skip:])
            df.append((key, kk, times, out))
    df = pd.DataFrame.from_records(df, columns=["ds_size", "case", "times", "sizes"])
    return df


def config_and_get_dtype_case(case, xnp):
    if case.find("double") >= 0:
        dtype = xnp.float64
    else:
        dtype = xnp.float32
    if case.find("cpu") >= 0:
        print("activating CPU")
        config.update('jax_platform_name', 'cpu')
    if dtype == xnp.float64:
        print("using double")
        config.update("jax_enable_x64", True)
    print(dtype)
    print(f"Running case: {case}")
    return dtype


def get_data_class1(N, xnp, dtype):
    size = int(N**0.5)
    diag1 = generate_spectrum(coeff=0.1, scale=1.0, size=size, dtype=np.float32)
    A1 = xnp.array(generate_pd_from_diag(diag1, dtype=np.float32, seed=48), dtype=dtype)
    A1 = SelfAdjoint(lazify(A1))
    A2 = Diagonal(generate_spectrum(coeff=1.0, scale=1.0, size=size, dtype=np.float32))
    clusters = [0.1, 0.3, 0.7, 0.85, 0.99]
    sizes = [size // 5 for _ in range(len(clusters))]
    diag2 = generate_clustered_spectrum(clusters, sizes, std=0.025, seed=48)
    A3 = SelfAdjoint(lazify(xnp.array(generate_diagonals(diag2, seed=21), dtype=dtype)))
    K = kron(A1, A2 @ A3)
    return K


def data_to_csr_neighbors(x, xnp, dtype, n_neighbors=10):
    adjacency = NearestNeighbors(n_neighbors=n_neighbors).fit(np.array(x))
    mode = "connectivity"
    out = adjacency.kneighbors_graph(adjacency._fit_X, mode=mode, n_neighbors=n_neighbors)
    data = xnp.array(out.data, dtype=dtype)
    indices = xnp.array(out.indices, dtype=xnp.int64)
    indptr = xnp.array(out.indptr, dtype=xnp.int64)
    return data, indices, indptr, out.shape


def data_to_csr(x, xnp, gamma):
    dist = compute_distances(x, xnp, gamma)
    # adjacency = make_adjacency_matrix(dist, xnp, threshold)
    adjacency = dist
    return transform_adjacency_to_csr(adjacency, xnp)


def transform_adjacency_to_csr(adjacency, xnp):
    out = csr_matrix(np.array(adjacency))
    data = xnp.array(out.data, dtype=adjacency.dtype)
    indices = xnp.array(out.indices, dtype=xnp.int64)
    indptr = xnp.array(out.indptr, dtype=xnp.int64)
    return data, indices, indptr, out.shape


def make_adjacency_matrix(dist, xnp, threshold):
    adjacency = xnp.where(dist > threshold, 0., 1.)
    return adjacency


def compute_distances(x, xnp, gamma=1.):
    dist_matrix = xnp.sum((x[None, :] - x[:, None])**2., axis=-1)
    dist = xnp.exp(-0.5 * gamma * dist_matrix)
    return dist


def print_time_taken(delta, text='Experiment took: ', logger=None):
    minutes = floor(delta / 60)
    seconds = delta - minutes * 60
    message = text + f'{minutes:4d} min and {seconds:4.2f} sec'
    if logger is not None:
        logger.info(message)
    else:
        print(message)


def load_object(filepath):
    with open(file=filepath, mode='rb') as f:
        obj = pickle.load(file=f)
    return obj


def save_object(obj, filepath, use_highest=True):
    os.makedirs(Path(filepath).parent, exist_ok=True)
    protocol = pickle.HIGHEST_PROTOCOL if use_highest else pickle.DEFAULT_PROTOCOL
    with open(file=filepath, mode='wb') as f:
        pickle.dump(obj=obj, file=f, protocol=protocol)

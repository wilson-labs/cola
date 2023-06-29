import time
import jax.numpy as jnp
import numpy as np
from cola.experiment_utils import print_time_taken, save_object
from cola.experiment_utils import construct_minimal_surface_setting
from cola.experiment_utils import get_times_minimal_surface
from cola.experiment_utils import get_times_minimal_scipy
from cola.experiment_utils import get_times_minimal_jax
# from jax.config import config
# config.update("jax_enable_x64", True)
# config.update('jax_platform_name', 'cpu')

save_output = True
Ns = [100, 50, 25, 10, 5]
repeat, results, tol = 2, {}, 5e-3
output_path = "./logs/minimal.pkl"

tic = time.time()
for N in Ns:
    results[N] = {}

    xgrid = jnp.linspace(-1, 1, N)
    xygrid = jnp.stack(jnp.meshgrid(xgrid, xgrid), axis=-1)
    dx = xgrid[1] - xgrid[0]
    x, y = xygrid.transpose((2, 0, 1))
    domain = (jnp.abs(x) < 1) & (jnp.abs(y) < 1)
    boundary_vals = np.zeros_like(x)
    boundary_vals[:, 0] = 1 - y[:, 0]**2
    boundary_vals[:, -1] = 1 - y[:, -1]**2

    J_matvec, pde_op = construct_minimal_surface_setting(dx, boundary_vals, domain)

    get_times_minimal_surface(J_matvec, pde_op, (domain, x), results[N], repeat, tol)
    get_times_minimal_scipy(J_matvec, pde_op, (domain, x), results[N], repeat, tol)
    get_times_minimal_jax(J_matvec, pde_op, (domain, x), results[N], repeat, tol)

toc = time.time()
print_time_taken(toc - tic)
if save_output:
    save_object(results, filepath=output_path)

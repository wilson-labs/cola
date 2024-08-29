from cola.ops import Dense, Diagonal, LinearOperator
from cola.utils.test_utils import get_xnp

op_names: set[str] = {
    "psd_diagonal"
    "nonsquare_dense",
}


def get_test_operator(backend: str, precision: str, op_name: str, device: str = 'cpu') -> LinearOperator:
    xnp = get_xnp(backend)
    dtype = getattr(xnp, precision)
    device = xnp.device(device)

    if backend == 'jax' and dtype == xnp.float64:
        import jax
        jax.config.update('jax_enable_x64', True)

    op = None
    match op_name.split('_', 1):
        case ("psd", "diagonal"):
            op = Diagonal(xnp.array([.1, .5, .22, 8.], dtype=dtype, device=device))

        case ("nonsquare", sub_op_name):
            M1 = xnp.array([[1., 0., 0.], [3., -4., 2.]], dtype=dtype, device=device)
            match sub_op_name:
                case 'dense':
                    op = Dense(M1)
    if op is None:
        raise ValueError(op_name)
    return op


def get_test_operators(backend: str, precision: str, device: str = 'cpu') -> list[LinearOperator]:
    return [
        get_test_operator(backend=backend, precision=precision, device=device, op_name=op_name) for op_name in op_names
    ]


__all__ = ["get_test_operators", "op_names"]

from functools import wraps


def slq_autograd(slq_bwd):
    def wrap_slq(slq_fn):
        @wraps(slq_fn)
        def slq_w_A_arg(A, *args, **kwargs):
            par, unflatten = A.flatten()

            def fwd(params):
                A = unflatten(params)
                output = slq_fn(A, *args, **kwargs)
                res = (params, )
                return output, res

            def bwd(res, d_ouputs):
                dA, *_ = slq_bwd(res, d_ouputs, unflatten, *args, **kwargs)
                return tuple(dA.flatten()[0])

            if A.ops.__name__.find('torch') >= 0:
                from torch.autograd import Function

                class SLQ(Function):
                    @staticmethod
                    def forward(ctx, *params):
                        output, res = fwd(params)
                        ctx.res = res
                        return output

                    @staticmethod
                    def backward(ctx, *grads):
                        return bwd(ctx.res, grads)

                return SLQ.apply(*par)
            elif A.ops.__name__.find('jax') >= 0:
                from jax import custom_vjp

                @custom_vjp
                def slq(params):
                    return fwd(params)[0]

                slq.defvjp(fwd, bwd)

                return slq(par)
            else:
                NotImplemented("Unknown ops for autograd")

        return slq_w_A_arg

    return wrap_slq


def solver_autograd(solver_bwd):
    def wrap_solver(solver_fn):
        @wraps(solver_fn)
        def solver_w_A_arg(A, *args, **kwargs):
            par, unflatten = A.flatten()

            def fwd(params):  # params -> y
                A = unflatten(params)
                output = solver_fn(A, *args, **kwargs)
                res = (params, output)
                return output, res

            def bwd(res, d_ouputs):  # dy -> dparams
                # TODO: gradients for all of solver
                dA, *_ = solver_bwd(res, d_ouputs, unflatten, *args, **kwargs)
                return tuple(dA.flatten()[0])

            if A.ops.__name__.find('torch') >= 0:
                from torch.autograd import Function

                class Solver(Function):
                    @staticmethod
                    def forward(ctx, *params):
                        output, res = fwd(params)
                        ctx.res = res
                        return output

                    @staticmethod
                    def backward(ctx, *grads):
                        return bwd(ctx.res, grads)

                return Solver.apply(*par)
            elif A.ops.__name__.find('jax') >= 0:
                from jax import custom_vjp

                @custom_vjp
                def solver(params):  # params -> y and has autograd
                    return fwd(params)[0]

                solver.defvjp(fwd, bwd)

                return solver(par)
            else:
                NotImplemented("Unknown ops for autograd")

        return solver_w_A_arg

    return wrap_solver

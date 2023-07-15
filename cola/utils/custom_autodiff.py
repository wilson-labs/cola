from functools import wraps


def iterative_autograd(iterative_bwd):
    def wrap_iterative(iterative_fn):
        @wraps(iterative_fn)
        def iterative_w_A_arg(A, *args, **kwargs):
            par, unflatten = A.flatten()

            def fwd(params):  # params -> y
                A = unflatten(params)
                output = iterative_fn(A, *args, **kwargs)
                res = (params, output)
                return output, res

            def bwd(res, d_ouputs):  # dy -> dparams
                # TODO: gradients for all of solver
                dA, *_ = iterative_bwd(res, d_ouputs, unflatten, *args, **kwargs)
                return tuple(dA.flatten()[0])

            if A.ops.__name__.find('torch') >= 0:
                from torch.autograd import Function

                class Iterative(Function):
                    @staticmethod
                    def forward(ctx, *params):
                        output, res = fwd(params)
                        ctx.res = res
                        return output

                    @staticmethod
                    def backward(ctx, *grads):
                        return bwd(ctx.res, grads)

                return Iterative.apply(*par)
            elif A.ops.__name__.find('jax') >= 0:
                from jax import custom_vjp

                @custom_vjp
                def iterative(params):  # params -> y and has autograd
                    return fwd(params)[0]

                iterative.defvjp(fwd, bwd)

                return iterative(par)
            else:
                NotImplemented("Unknown ops for autograd")

        return iterative_w_A_arg

    return wrap_iterative

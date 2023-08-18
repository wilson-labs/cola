from functools import wraps


def iterative_autograd(iterative_bwd):
    """ Autograd wrapper for iterative solvers like CG, Lanczos, SLQ, etc.
        Will construct a custom autograd rule for the iterative solver on every call
        of the function.

        Args:
            iterative_bwd (Callable): The vjp for the iterative solver.
            Should map ((params, output), d_output, unflatten, *args, **kwargs) -> d_A
            (Shouldn't it be (d_output, *args, **kwargs) -> d_A?) )
        
        Usage:
        
        @iterative_autograd(iterative_bwd)
        def iterative_fwd(A, *args, **kwargs):
            # do stuff with A
            # return output

        """
    def wrap_iterative(iterative_fn):
        """ Inner level wrapper """
        @wraps(iterative_fn)
        def iterative_w_A_arg(A, *args, **kwargs):
            """ Version of iterative_fn that takes A as first argument and has a custom autograd rule."""
            par, unflatten = A.flatten()

            # construct a fwd function that maps params -> (output, (params, output))
            def fwd(params):  # params -> y
                A = unflatten(params)
                output = iterative_fn(A, *args, **kwargs)
                res = (params, output)
                return output, res

            # construct a bwd function that maps (params, output), doutput -> dparams
            def bwd(res, d_ouputs):  # dy -> dparams
                # TODO: gradients for all of solver
                dA, *_ = iterative_bwd(res, d_ouputs, unflatten, *args, **kwargs)
                return (type(par)(dA.flatten()[0]),)

            if A.xnp.__name__.find('torch') >= 0:
                from torch.autograd import Function

                class Iterative(Function):
                    @staticmethod
                    def forward(ctx, *params):
                        output, res = fwd(params)
                        ctx.res = res
                        return output

                    @staticmethod
                    def backward(ctx, *grads):
                        #return bwd(ctx.res, grads)
                        return tuple(bwd(ctx.res, grads)[0])

                return Iterative.apply(*par)
            elif A.xnp.__name__.find('jax') >= 0:
                from jax import custom_vjp

                @custom_vjp
                def iterative(params):  # params -> y and has autograd
                    return fwd(params)[0]
                #jax_bwd = lambda *args,**kwargs: ((bwd(*args,**kwargs)),)
                iterative.defvjp(fwd, bwd)

                return iterative(par)
            else:
                NotImplemented("Unknown ops for autograd")

        return iterative_w_A_arg

    return wrap_iterative

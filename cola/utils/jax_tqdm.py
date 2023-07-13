# Credit to Jeremie Coullon
# Adapted from https://github.com/jeremiecoullon/jax-tqdm

import typing

import jax
from jax.experimental import host_callback
from tqdm.auto import tqdm
import functools
import numpy as np
import time


def scan_tqdm(n: int, message: typing.Optional[str] = None) -> typing.Callable:
    """
    tqdm progress bar for a JAX scan
    Parameters
    ----------
    n : int
        Number of scan steps/iterations.
    message : str
        Optional string to prepend to tqdm progress bar.
    Returns
    -------
    typing.Callable:
        Progress bar wrapping function.
    """

    _update_progress_bar, close_tqdm = build_tqdm(n, message)

    def _scan_tqdm(func):
        """Decorator that adds a tqdm progress bar to `body_fun` used in `jax.lax.scan`.
        Note that `body_fun` must either be looping over `jnp.arange(n)`,
        or be looping over a tuple who's first element is `jnp.arange(n)`
        This means that `iter_num` is the current iteration number
        """
        def wrapper_progress_bar(carry, x):
            if type(x) is tuple:
                iter_num, *_ = x
            else:
                iter_num = x
            _update_progress_bar(iter_num)
            result = func(carry, x)
            return close_tqdm(result, iter_num)

        return wrapper_progress_bar

    return _scan_tqdm


def loop_tqdm(n: int, message: typing.Optional[str] = None) -> typing.Callable:
    """
    tqdm progress bar for a JAX fori_loop
    Parameters
    ----------
    n : int
        Number of iterations.
    message : str
        Optional string to prepend to tqdm progress bar.
    Returns
    -------
    typing.Callable:
        Progress bar wrapping function.
    """

    _update_progress_bar, close_tqdm = build_tqdm(n, message)

    def _loop_tqdm(func):
        """
        Decorator that adds a tqdm progress bar to `body_fun`
        used in `jax.lax.fori_loop`.
        """
        def wrapper_progress_bar(i, val):
            _update_progress_bar(i)
            result = func(i, val)
            return close_tqdm(result, i)

        return wrapper_progress_bar

    return _loop_tqdm


def build_tqdm(
        n: int,
        message: typing.Optional[str] = None) -> typing.Tuple[typing.Callable, typing.Callable]:
    """
    Build the tqdm progress bar on the host
    """

    if message is None:
        message = f"Running for {n:,} iterations"
    tqdm_bars = {}

    if n > 20:
        print_rate = int(n / 20)
    else:
        print_rate = 1
    remainder = n % print_rate

    def _define_tqdm(arg, transform):
        tqdm_bars[0] = tqdm(range(n))
        tqdm_bars[0].set_description(message, refresh=False)

    def _update_tqdm(arg, transform):
        tqdm_bars[0].update(arg)

    def _update_progress_bar(iter_num):
        "Updates tqdm from a JAX scan or loop"
        _ = jax.jax.lax.cond(
            iter_num == 0,
            lambda _: host_callback.id_tap(_define_tqdm, None, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

        _ = jax.lax.cond(
            # update tqdm every multiple of `print_rate` except at the end
            (iter_num % print_rate == 0) & (iter_num != n - remainder),
            lambda _: host_callback.id_tap(_update_tqdm, print_rate, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

        _ = jax.lax.cond(
            # update tqdm by `remainder`
            iter_num == n - remainder,
            lambda _: host_callback.id_tap(_update_tqdm, remainder, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

    def _close_tqdm(arg, transform):
        tqdm_bars[0].close()

    def close_tqdm(result, iter_num):
        return jax.lax.cond(
            iter_num == n - 1,
            lambda _: host_callback.id_tap(_close_tqdm, None, result=result),
            lambda _: result,
            operand=None,
        )

    return _update_progress_bar, close_tqdm


# def _body_pbar(errorfn, tol, desc='', every=1, hide=False):
#     if hide: return lambda body: body

#     def decorated_body(body):
#         info = {'progval': 0, 'count': 0, 'pbar': None}
#         default_desc = f"Running {body.__name__}"

#         def construct_tqdm(arg, transform):
#             print(f'constructing pbar')
#             if info['pbar'] is None:
#                 info['pbar'] = tqdm(
#                     total=100, desc=f'{desc or default_desc}', bar_format=
#                     "{l_bar}{bar}| {n:.3g}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
#                 )

#         def update_tqdm(arg, transform):
#             error = errorfn(arg)
#             errstart = info.setdefault('errstart', error)
#             progress = max(
#                 100 * np.log(error / errstart) / np.log(tol / errstart) - info['progval'], 0)
#             progress = min(100 - info['progval'], progress)
#             if progress > 0:
#                 info['progval'] += progress
#                 info['pbar'].update(progress)
#             if error < tol:
#                 info['pbar'].close()

#         def increment_count(arg, transform):
#             info['count'] += 1
#             print(info['count'])

#         @functools.wraps(body)
#         def newbody(val):
#             _ = jax.lax.cond(
#                 info['count'] == 0,
#                 lambda _: host_callback.id_tap(construct_tqdm, None, result=info['count']),
#                 lambda _: info['count'],
#                 operand=None,
#             )
#             nextval = body(val)

#             _ = jax.lax.cond(
#                 info['count'] % every == 0,
#                 lambda _: host_callback.id_tap(update_tqdm, val, result=info['count']),
#                 lambda _: info['count'],
#                 operand=None,
#             )
#             _ = host_callback.id_tap(increment_count, None, result=info['count'])
#             return nextval

#         return newbody

#     return decorated_body

# while loop with progress bar


def pbar_while(errorfn, tol, desc='', every=1, hide=False):
    """ Decorator for while loop with progress bar. Assumes that
        errorfn is a function of the loop variable and returns a scalar
        that starts at a given value and decreases to tol as the loop progresses."""
    if hide:
        return jax.lax.while_loop

    def new_while(cond_fun, body_fun, init_val):
        info = {'progval': 0, 'pbar': None}
        default_desc = f"Running {body_fun.__name__}"

        def construct_tqdm(arg, transform):
            info['pbar'] = tqdm(
                total=100, desc=f'{desc or default_desc}', bar_format=
                "{l_bar}{bar}| {n:.3g}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")

        def update_tqdm(arg, transform):
            error = errorfn(arg)
            errstart = info.setdefault('errstart', error)
            progress = max(
                100 * np.log(error / errstart) / np.log(tol / errstart) - info['progval'], 0)
            progress = min(100 - info['progval'], progress)
            if progress > 0:
                info['progval'] += progress
                info['pbar'].update(progress)

        def close_tqdm(arg, transform):
            update_tqdm(arg, transform)
            info['pbar'].close()

        def newbody(ival):
            i, val = ival
            jax.lax.cond(
                i % every == 0,
                lambda _: host_callback.id_tap(update_tqdm, val, result=i),
                lambda _: i,
                operand=None,
            )
            return (i + 1, body_fun(val))

        def newcond(ival):
            i, val = ival
            out = jax.lax.cond(cond_fun(val), lambda _: True,
                               lambda _: host_callback.id_tap(close_tqdm, val, result=False),
                               operand=None)
            return out

        host_callback.id_tap(construct_tqdm, None)
        _, val = jax.lax.while_loop(newcond, newbody, (0, init_val))
        return val

    return new_while


def while_loop_winfo(errorfn, tol, every=1, desc='', pbar=False, **kwargs):
    """ Decorator for while loop with progress bar. 
    
        Assumes that errorfn is a function of the loop variable and returns a scalar
        that starts at a given value and decreases to tol as the loop progresses.
        
        Args:
            errorfn: function of the while state that returns a scalar tracking the error (e.g. residual)
            tol: tolerance for errorfn
            every: update progress bar every this many iterations
            desc: description for progress bar
            pbar: whether to show progress bar
            kwargs: additional info to pass to progress bar

        Returns: (tuple) while_loop, info_dict
        """
    info = {}

    def new_while(cond_fun, body_fun, init_val):

        default_desc = f"Running {body_fun.__name__}"

        def construct_info(*_):
            info.pop('errstart', None)
            info.update({'progval': 0, 'errors': [], **kwargs})
            info['iteration_time'] = time.time()
            if pbar:
                bar_format = "{l_bar}{bar}| {n:.3g}/{total_fmt} [{elapsed}<{remaining},"
                bar_format += "{rate_fmt}{postfix}]"
                info['pbar'] = tqdm(total=100, desc=f'{desc or default_desc}',
                                    bar_format=bar_format)

        def update_info(arg, _):
            error = errorfn(arg)
            info['errors'].append(error)
            if pbar:
                errstart = info.setdefault('errstart', error)
                progress = max(
                    100 * np.log(error / errstart) / np.log(tol / errstart) - info['progval'], 0)
                progress = min(100 - info['progval'], progress)
                if progress > 0:
                    info['progval'] += progress
                    info['pbar'].update(progress)

        def close_info(arg, transform):
            i, val = arg
            update_info(val, transform)
            info['iteration_time'] = (time.time() - info['iteration_time']) / (i + 1)
            if pbar:
                info['pbar'].close()
                info.pop('errstart')
                info.pop('pbar')
            info.pop('progval')
            info['errors'] = np.array(info['errors'][1:])
            info['iterations'] = i + 1

        def newbody(ival):
            i, val = ival
            jax.lax.cond(
                i % every == 0,
                lambda _: host_callback.id_tap(update_info, val, result=i),
                lambda _: i,
                operand=None,
            )
            return (i + 1, body_fun(val))

        def newcond(ival):
            _, val = ival
            out = jax.lax.cond(cond_fun(val), lambda _: True,
                               lambda _: False,#host_callback.id_tap(close_info, ival, result=False),
                               operand=None)
            return out

        host_callback.id_tap(construct_info, None)
        k, val = jax.lax.while_loop(newcond, newbody, (0, init_val))
        host_callback.id_tap(close_info,(k,val))
        return val

    return new_while, info

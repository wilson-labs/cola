import functools
import time
import numpy as np
from torch import Tensor
from tqdm.auto import tqdm


def while_loop_winfo(errorfn, tol, every=1, desc='', pbar=False, **kwargs):
    """ Decorator for while loop with progress bar. 
    
    Assumes that
    errorfn is a function of the loop variable and returns a scalar
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
        info.update({'progval': 0, 'iterations': 0, 'errors': [], **kwargs})
        default_desc = f"Running {body_fun.__name__}"
        if pbar:
            bar_format = "{l_bar}{bar}| {n:.3g}/{total_fmt} [{elapsed}<{remaining}, "
            bar_format += "{rate_fmt}{postfix}]"
            info['pbar'] = tqdm(total=100, desc=f'{desc or default_desc}', bar_format=bar_format)

        def newcond(state):
            if info['iterations'] == 0:
                info['iteration_time'] = time.time()
            error = errorfn(state)
            if isinstance(error, Tensor):
                error = error.cpu().data.item()
            if not info['iterations'] % every:
                info['errors'].append(error)
                if pbar:
                    update_pbar(error, tol, info)
            info['iterations'] += 1
            return cond_fun(state)

        out = while_loop(newcond, body_fun, init_val)
        info['iteration_time'] = (time.time() - info['iteration_time']) / info['iterations']
        error = errorfn(out)
        if isinstance(error, Tensor):
            error = error.cpu().data.item()
        info['errors'].append(error)
        info['errors'] = np.array(info['errors'][2:])
        if pbar:
            update_pbar(info['errors'][-1], tol, info)
            info['pbar'].close()
            info.pop('errstart')
            info.pop('pbar')
        info.pop('progval')
        return out

    return new_while, info


def update_pbar(error, tol, info):
    errstart = info.setdefault('errstart', error)
    progress = max(100 * np.log(error / errstart) / np.log(tol / errstart) - info['progval'], 0)
    progress = min(100 - info['progval'], progress)
    if progress > 0:
        info['progval'] += progress
        info['pbar'].update(progress)


def while_loop(cond_fun, body_fun, init_val):
    val = init_val
    while cond_fun(val):
        val = body_fun(val)
    return val


def pbar_while(errorfn, tol, desc='', every=1, hide=False):
    """ Decorator for while loop with progress bar. Assumes that
        errorfn is a function of the loop variable and returns a scalar
        that starts at a given value and decreases to tol as the loop progresses."""
    def new_while(cond_fun, body_fun, init_val):
        newbody = body_pbar(errorfn, tol, desc, every, hide)(body_fun)
        val = init_val
        while cond_fun(val):
            val = newbody(val)
        return val

    return new_while


def body_pbar(errorfn, tol, desc='', every=1, hide=False):
    if hide:
        return lambda body: body

    def decorated_body(body):
        info = {'progval': 0, 'count': 0, 'pbar': None}
        default_desc = f"Running {body.__name__}"

        @functools.wraps(body)
        def wrapper(*args, **kwargs):
            if info['pbar'] is None:
                _bar_format = "{l_bar}{bar}| {n:.3g}/{total_fmt} [{elapsed}<{remaining},"
                _bar_format += " {rate_fmt}{postfix}]"
                info['pbar'] = tqdm(total=100, desc=f'{desc or default_desc}',
                                    bar_format=_bar_format)
            val = body(*args, **kwargs)
            info['count'] += 1
            if info['count'] % every != 0:
                return val
            error = errorfn(val)
            if isinstance(error, Tensor):
                error = error.cpu().data.item()
            errstart = info.setdefault('errstart', error)
            progress = max(
                100 * np.log(error / errstart) / np.log(tol / errstart) - info['progval'], 0)
            progress = min(100 - info['progval'], progress)
            if progress > 0:
                info['progval'] += progress
                info['pbar'].update(progress)
            if error < tol:
                info['pbar'].close()
            return val

        return wrapper

    return decorated_body

import functools
import time
import numpy as np

from tqdm.auto import tqdm


def while_loop_winfo(errorfn, tol, max_iters=None, every=1, desc='', pbar=False, **kwargs):
    """ Decorator for while loop with progress bar.

    Assumes that
    errorfn is a function of the loop variable and returns a scalar
    that starts at a given value and decreases to tol as the loop progresses.

    Args:
        errorfn: function of the while state that returns a scalar tracking the error
         (e.g. residual)
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
            try:
                from torch import Tensor
                if isinstance(error, Tensor):
                    error = error.cpu().data.item()
            except ImportError:
                pass
            if not (info['iterations'] % every):
                info['errors'].append(error)
                if pbar:
                    update_pbar(error, tol, info, max_iters)
            info['iterations'] += 1
            return cond_fun(state)

        out = while_loop(newcond, body_fun, init_val)
        info['iteration_time'] = (time.time() - info['iteration_time']) / info['iterations']
        error = errorfn(out)
        try:
            from torch import Tensor
            if isinstance(error, Tensor):
                error = error.cpu().data.item()
        except ImportError:
            pass
        info['errors'].append(error)
        info['errors'] = np.array(info['errors'][2:])
        if pbar:
            update_pbar(info['errors'][-1], tol, info, max_iters)
            info['pbar'].close()
            info.pop('errstart')
            info.pop('pbar')
        info.pop('progval')
        return out

    return new_while, info


def update_pbar(error, tol, info, max_iters):
    errstart = info.setdefault('errstart', error)
    howclose = np.log(error / errstart) / np.log(tol / errstart)
    if max_iters is not None:
        howclose = max(info['iterations'] / max_iters, howclose)
    progress = min(100 - info['progval'], max(100 * howclose - info['progval'], 0))
    if progress > 0:
        info['progval'] += progress
        info['pbar'].update(progress)

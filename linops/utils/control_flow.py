def for_loop(lower, upper, body_fun, init_val):
    state = init_val
    for iter in range(lower, upper):
        state = body_fun(iter, state)
    return state


def while_loop(cond_fun, body_fun, init_val):
    state = init_val
    while cond_fun(state):
        state = body_fun(state)
    return state

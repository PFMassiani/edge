import torch
from inspect import signature
from decorator import decorator


def ensure_tensor(x):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    return x


def tensorwrap(*deco_args):
    wrap_all = False
    wrap_args = False
    wrap_names = False
    num_params = len(deco_args)
    if num_params == 0:
        wrap_all = True
    elif num_params == 1 and isinstance(deco_args[0], int):
        wrap_end = deco_args[0]
        wrap_args = True
    else:
        names_to_wrap = deco_args
        wrap_names = True

    @decorator
    def numpy_tensor_wrapper(func, *args, **kwargs):
        args = list(args)
        all_names = list(signature(func).parameters.keys())
        wrap_start = 1 if all_names[0] == 'self' else 0

        if wrap_all:
            for argnum in range(wrap_start, len(args)):
                args[argnum] = ensure_tensor(args[argnum])
            for name, arg in kwargs.items():
                kwargs[name] = ensure_tensor(arg)
            return func(*args, **kwargs)

        elif wrap_args:
            nonlocal wrap_end
            wrap_end += wrap_start
            for argnum in range(wrap_start, wrap_end):
                args[argnum] = ensure_tensor(args[argnum])
            return func(*args, **kwargs)

        elif wrap_names:
            kwargs.update(dict(zip(
                all_names[:len(args)],
                args
            )))
            for name in names_to_wrap:
                kwargs[name] = ensure_tensor(kwargs[name])
            return func(**kwargs)

    return numpy_tensor_wrapper

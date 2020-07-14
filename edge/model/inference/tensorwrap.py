import torch
from inspect import signature
from decorator import decorator


def ensure_tensor(x):
    if not torch.is_tensor(x):
        x = torch.tensor(x).float()
    return x


def tensorwrap(*deco_args):
    """
    Method decorator. Preprocesses some of the arguments of the decorated method to cast them to torch.Tensor.
    Can be called in three fashions:
        * no parameters: all parameters are cast to torch.Tensor
        * one integer parameter, k: the first k arguments are cast to torch.Tensor
        * one or many string parameters: the parameters with the corresponding names are cast to torch.Tensor
    In the first two cases, the first parameter of the method is systematically left untouched if it is called `self`.
    In the first case, parentheses must still be used.
    :param deco_args: either nothing, an integer, or several strings
    :return: the method with automatic torch.Tensor cast of some of its parameters
    """
    # We need to do some parsing of deco_args to figure out what its values represent
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

    # This decorator indicates that the following function is a decorator, and provides nice tools such as preserving
    # the name and the docstring of the decorated function
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
            local_wrap_end = wrap_end
            local_wrap_end += wrap_start
            for argnum in range(wrap_start, local_wrap_end):
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

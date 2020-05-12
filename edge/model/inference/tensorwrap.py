import torch
from inspect import signature
from functools import wraps


def tensorwrap(*deco_args):
    called_on_func = False
    if len(deco_args) == 1 and callable(deco_args[0]):
        wrap_end = None
        retrieve_parameter_names = True
        called_on_func = True
    elif len(deco_args) == 1 and isinstance(deco_args[0], int):
        wrap_end = deco_args[0]
        retrieve_parameter_names = True
    else:
        parameters_to_wrap = deco_args
        wrap_end = len(deco_args)
        retrieve_parameter_names = False

    def tensor_wrapper_decorator(func):
        nonlocal wrap_end
        nonlocal parameters_to_wrap
        all_parameter_names = list(signature(func).parameters.keys())
        wrap_start = 1 if all_parameter_names[0] == 'self' else 0
        wrap_end = wrap_end if wrap_end is None else wrap_end + wrap_start
        if retrieve_parameter_names:
            parameters_to_wrap = all_parameter_names[wrap_start:wrap_end]

        @wraps(func)
        def numpy_tensor_wrapper(*args, **kwargs):
            kwargs.update(dict(zip(
                all_parameter_names[:len(args)],
                args
            )))
            for pname in parameters_to_wrap:
                if not torch.is_tensor(kwargs[pname]):
                    kwargs[pname] = torch.from_numpy(kwargs[pname])

            return func(**kwargs)

    if called_on_func:
        return tensor_wrapper_decorator(deco_args[0])
    else:
        return tensor_wrapper_decorator

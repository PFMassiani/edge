def bind(instance, func, as_name=None):
    """
    Binds the function `func` to `instance`, with name `as_name` if provided, or func.__name__ otherwise.
    The function `func` should accept the instance as the first argument, i.e. `self`.
    :param instance: the instance to which the function should be bound
    :param func: the function to bind
    :param as_name: the name of the bound function
    :return: the bound function
    """
    if as_name is None:
        as_name = func.__name__
    bound_method = func.__get__(instance, instance.__class__)
    setattr(instance, as_name, bound_method)
    return bound_method

from numbers import Number


def get_terminal_and_direction_from_args(args):
    if len(args) == 1:
        terminal = False
        direction = 0
        if isinstance(args[0], bool):
            terminal = args[0]
        elif isinstance(args[0], Number):
            direction = args[0]
        else:
            raise TypeError("Expected type `bool` or `numbers.Number`, got "
                            f"{type(args[0])} instead")
    elif len(args) == 2:
        terminal = None
        direction = None
        for a in args:
            if isinstance(a, bool):
                terminal = a
            elif isinstance(a, Number):
                direction = a
            else:
                raise TypeError("Expected type `bool` or `numbers.Number`, got"
                                f" {type(a)} instead")
        if (terminal is None) or (direction is None):
            raise TypeError("Expected types `bool` and `numbers.Number`, but "
                            "one is missing")
    else:
        raise ValueError("Too many parameters: expected at most 2, got "
                         f"{len(args)}")
    return terminal, direction


def event(*args):
    called_on_evt_func = False
    if len(args) > 2:
        raise ValueError("Too many parameters: expected at most 2, got "
                         f"{len(args)}")
    elif len(args) == 0:
        terminal = False
        direction = 0
    elif callable(args[0]):
        terminal = False
        direction = 0
        evt_func = args[0]
        called_on_evt_func = True
    else:
        term_dir = get_terminal_and_direction_from_args(args)
        terminal, direction = term_dir

    def evt_decorator(func):
        func.terminal = terminal
        func.direction = direction
        func.is_event = True
        return func

    if called_on_evt_func:
        return evt_decorator(evt_func)
    else:
        return evt_decorator


class EventBased:
    def get_events(self):
        methods = [getattr(self, f)
                   for f in dir(self)
                   if callable(getattr(self, f)) and not f.startswith("__")
                   ]
        evt_methods = [m
                       for m in methods
                       if hasattr(m, 'is_event') and getattr(m, 'is_event')
                       ]
        return evt_methods

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


class event:
    """
    Decorates a function :
        * adds a `terminal` attribute
        * adds a `direction` attribute
        * adds a `is_event` attribute (used by the class decorator
            `event_based`)
        * sets it as a static function, so the first parameter needs not to
            be an instance of the class. This is useful for the integration
            with the scipy.integrate module, for which events need to have
            exactly two parameters

    """
    # Adapted from
    # https://stackoverflow.com/questions/1697501/staticmethod-with-property
    def __init__(self, *args):
        if len(args) > 2:
            raise ValueError("Too many parameters: expected at most 2, got "
                             f"{len(args)}")
        elif callable(args[0]):
            self.terminal = False
            self.direction = 0
            self.evt_func = args[0]
            self._set_func_attributes()
        else:
            term_dir = get_terminal_and_direction_from_args(args)
            self.terminal, self.direction = term_dir
            self.evt_func = None

    def _set_func_attributes(self):
        self.evt_func.is_event = True
        self.evt_func.terminal = self.terminal
        self.evt_func.direction = self.direction

    def __call__(self, evt_func):
        if self.evt_func is not None:
            raise ValueError('evt_funct has already been initialized')
        self.evt_func = evt_func
        self._set_func_attributes()

        return self

    def __get__(self, cls, owner):
        return staticmethod(self.evt_func).__get__(None, owner)


def event_based(cls):
    methods = [getattr(cls, func)
               for func in dir(cls)
               if callable(getattr(cls, func)) and not func.startswith("__")
               ]
    events = []
    for method in methods:
        if hasattr(method, 'is_event') and method.is_event:
            events.append(method)

    def get_events(clsinstance):
        return events

    setattr(cls, 'get_events', get_events)
    return cls

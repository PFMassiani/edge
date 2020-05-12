from gpytorch import constraints
from numbers import Number


def atleast_2d(ten):
    if ten.ndimension() == 1:
        ten = ten.unsqueeze(-1)
    return ten


def constraint_from_tuple(bounds, default=None):
    if bounds is None:
        if default is None:
            return constraints.Positive()
        else:
            return default
    elif isinstance(bounds, Number):
        return constraints.GreaterThan(max(0., float(bounds)))
    else:
        lbound = max(0., float(bounds[0]))
        ubound = max(0., float(bounds[1]))
        return constraints.Interval(lbound, ubound)

from gpytorch import constraints
from numbers import Number
import numpy as np


def atleast_2d(ten):
    if ten.ndimension() == 1:
        ten = ten.unsqueeze(0)
    return ten


def constraint_from_tuple(bounds, default=None):
    if bounds is None:
        return default
    elif isinstance(bounds, Number):
        return constraints.GreaterThan(max(0., float(bounds)))
    else:
        lbound = max(0., float(bounds[0]))
        ubound = max(0., float(bounds[1]))
        return constraints.Interval(lbound, ubound)


def dynamically_import(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def get_hyperparameters(gp, constraints=False, around=None):
    params = {}
    for name, param, constraint \
            in gp.named_parameters_and_constraints():
        if constraint is not None:
            transformed = constraint.transform(param).cpu().detach().numpy().\
                          squeeze()
        else:
            transformed = param.cpu().detach().numpy().squeeze()
        if around is not None:
            transformed = np.around(transformed, around=around)
        entry = transformed if not constraints else (transformed, constraint)
        key = ''.join(name.split('raw_'))
        params[key] = entry
    return params

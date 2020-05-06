class OutOfSpace(Exception):
    """
    Raised when the index of a point in a DiscreteSpace is queried and the
    point is not in the set
    """


class InvalidTarget(Exception):
    """
    Raised when the target space for a projection is invalid
    """

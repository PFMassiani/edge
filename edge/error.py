class OutOfSpace(Exception):
    """
    Raised when the index of a point is queried and the point is not in the set
    """


class NotOnGrid(Exception):
    """
    Raised when the index of a point is queried with around_ok set to False and
    the point is not on the grid
    """


class InvalidTarget(Exception):
    """
    Raised when the target space for a projection is invalid
    """

from collections.abc import Iterable


def normalize_to_list(value):
    """
    Normalize a string or sequence of values into a list.

    Parameters
    ----------
    value : str, sequence, or None
        Input value to normalize. If a string is provided, it is wrapped in a
        single-element list. If a sequence is provided, it is converted to a list.
        If None, None is returned unchanged.

    Returns
    -------
    list or None
        A list representation of the input value, or None if the input was None.

    Raises
    ------
    TypeError
        If `value` is not None, a string, or a sequence.
    """

    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    raise TypeError(f"Input must be a string or a sequence of strings")

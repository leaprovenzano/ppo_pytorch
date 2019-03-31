

def inherit_docstring(obj: object) -> object:
    """decorator to which will update docstring with additional information about 
    first parent in mro ( if this object indeed has a parent other than object).


    Args:
        obj (object): class being instantiated

    Returns:
        object: class decorated with updated docstring 
    """
    if len(obj.mro()) > 2:
        if not obj.__doc__:
            obj.__doc__ = """\n"""

        obj.__doc__ = "{obj.__doc__}\nInherits From :\n\n\t{parent.__name__}:\n\t{parent.__doc__}".format(obj=obj, parent=obj.mro()[1])
    return obj

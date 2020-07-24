import importlib.util
import os
import sys


def static_vars(**kwargs):
    r'''Decorator to add :py:attr:`kwargs` as static vars to 'func'

        Example:

            .. code-block:: python

                >>> @static_vars(counter=0)
                ... def myfync():
                ...     myfync.counter += 1
                ...     return myfync.counter
                ...
                >>> print(myfunc())
                1
                >>> print(myfunc())
                2
                >>> print(myfunc())
                3
                >>> myfunc.counter = 100
                >>> print(myfunc())
                101
    '''
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


def import_module_from_file(file_path, module_name):
    assert isinstance(file_path, str) and os.path.exists(file_path),\
        "'file_path' must be a full path string with the python file to load"
    assert isinstance(module_name, str) and module_name,\
        "'module_name' must be a string with the python module name to load"

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

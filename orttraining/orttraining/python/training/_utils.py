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

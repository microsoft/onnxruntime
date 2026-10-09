# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
from contextlib import contextmanager


@contextmanager
def scoped_env_var(name: str, value: str):
    """Temporarily set an environment variable, restoring the previous value on exit.

    Keeps tests order-independent by ensuring env-var mutations do not leak into
    later tests running in the same process.
    """
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------


class FakeContext:
    """A mock up class used to represent ctx in unsfafe mode run.
    The reason we need ctx to be Python class is: users could assign any attribute to ctx.
    """

    def __init__(self):
        pass

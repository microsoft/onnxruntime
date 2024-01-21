#  // Copyright (c) Microsoft Corporation. All rights reserved.
#  // Licensed under the MIT License.
class BaseError(Exception):
    """Base class for errors originating from build.py."""

    pass


class BuildError(BaseError):
    """Error from running build steps."""

    def __init__(self, *messages):
        super().__init__("\n".join(messages))


class UsageError(BaseError):
    """Usage related error."""

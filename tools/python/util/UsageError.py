#  // Copyright (c) Microsoft Corporation. All rights reserved.
#  // Licensed under the MIT License.
import BaseError


class UsageError(BaseError):
    """Usage related error."""

    def __init__(self, message):
        super().__init__(message)
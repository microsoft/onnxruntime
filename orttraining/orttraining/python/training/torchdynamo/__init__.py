# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from typing import Set

# set of custom ops supported in DORT.
# It can contain, for example, `aten::custom_add`.
custom_symbols: Set[str] = set()

# register custom ops in DORT
def register_custom_op_in_dort(custom_op_name: str):
    custom_symbols.add(custom_op_name)

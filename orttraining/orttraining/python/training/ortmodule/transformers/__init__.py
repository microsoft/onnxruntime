# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os

_all_transformers = []

if "ORTMODULE_USE_EFFICIENT_ATTENTION" in os.environ and int(os.getenv("ORTMODULE_USE_EFFICIENT_ATTENTION")) == 1:
    from ._aten_attn import transform_aten_efficient_attention  # noqa: F401

    _all_transformers.append("transform_aten_efficient_attention")

__all__ = _all_transformers  # noqa: PLE0605

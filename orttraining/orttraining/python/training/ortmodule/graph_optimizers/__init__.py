# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os

_all_optimizers = []

if "ORTMODULE_USE_EFFICIENT_ATTENTION" in os.environ and int(os.getenv("ORTMODULE_USE_EFFICIENT_ATTENTION")) == 1:
    from ._aten_attn import optimize_graph_for_aten_efficient_attention  # noqa: F401

    _all_optimizers.append("optimize_graph_for_aten_efficient_attention")

__all__ = _all_optimizers  # noqa: PLE0605

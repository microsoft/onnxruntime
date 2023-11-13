# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os

from ._mm import triton_gemm, triton_gemm_out, triton_matmul, triton_matmul_out  # noqa: F401
from ._slice_scel import slice_scel, slice_scel_backward  # noqa: F401

_all_kernels = [
    "triton_gemm",
    "triton_gemm_out",
    "triton_matmul",
    "triton_matmul_out",
    "slice_scel",
    "slice_scel_backward",
]

if "ORTMODULE_USE_FLASH_ATTENTION" in os.environ and int(os.getenv("ORTMODULE_USE_FLASH_ATTENTION")) == 1:
    from ._flash_attn import flash_attn_backward, flash_attn_forward  # noqa: F401

    _all_kernels.extend(["flash_attn_forward", "flash_attn_backward"])

__all__ = _all_kernels  # noqa: PLE0605

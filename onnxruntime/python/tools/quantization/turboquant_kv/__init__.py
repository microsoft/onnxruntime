# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""TurboQuant KV cache quantization for ONNX Runtime.

Reference implementation in pure NumPy. Used for:
  1. Algorithm validation against the TurboQuant paper.
  2. Cross-checking the CUDA / WebGPU kernels.
  3. Generating the static Lloyd-Max codebooks shipped with the runtime.

The runtime kernels live in:
  - contrib_ops/cuda/bert/group_query_attention_turboquant.cuh (CUDA)
  - contrib_ops/webgpu/bert/flash_attention_*_turboquant.wgsl.template (WebGPU)
"""

from .centroids import (
    solve_lloyd_max,
    get_centroids,
    get_boundaries,
)
from .hadamard import (
    walsh_hadamard,
    rotate,
)
from .packing import (
    pack_3bit,
    unpack_3bit,
    pack_4bit,
    unpack_4bit,
    packed_size_bytes,
)
from .quantizer import (
    TurboQuantConfig,
    TQ_PRESETS,
    encode_keys,
    decode_keys,
    encode_values,
    decode_values,
    score_in_rotated_space,
)

__all__ = [
    "TurboQuantConfig",
    "TQ_PRESETS",
    "solve_lloyd_max",
    "get_centroids",
    "get_boundaries",
    "walsh_hadamard",
    "rotate",
    "pack_3bit",
    "unpack_3bit",
    "pack_4bit",
    "unpack_4bit",
    "packed_size_bytes",
    "encode_keys",
    "decode_keys",
    "encode_values",
    "decode_values",
    "score_in_rotated_space",
]

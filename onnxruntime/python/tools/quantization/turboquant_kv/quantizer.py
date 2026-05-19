# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""TurboQuant K/V encode and decode reference implementation (NumPy).

This is the canonical CPU reference. The CUDA / WebGPU kernels in
contrib_ops/{cuda,webgpu}/bert/ must produce numerically identical output
(up to fp16 rounding) for any input.

Pipeline for keys (per token, per head):
  1. Compute norm: n = ||k||
  2. Normalize:   x_hat = k / n
  3. Rotate:      y = x_hat @ H
  4. Quantize:    idx[j] = nearest_centroid_index(y[j])
  5. Pack:        packed = bit_pack(idx, bits)
  Storage: (packed, n) — n stored as fp16.

Pipeline for values (per token, per head):
  1. Compute scale + zero: scale = (max - min) / (2^bits - 1), zero = min
  2. Quantize:    idx[j] = round((v[j] - zero) / scale), clamped to [0, 2^bits - 1]
  3. Pack:        packed = bit_pack(idx, bits)
  Storage: (packed, scale, zero) — scale and zero stored as fp16.

Decode for keys:
  1. Unpack:      idx = bit_unpack(packed)
  2. Lookup:      y_hat = centroids[idx]
  3. (optional norm correction): y_hat = y_hat / ||y_hat|| * (correction factor)
  4. Denormalize: k_hat = n * y_hat
  IMPORTANT: k_hat is in the ROTATED space. Attention scoring is done in this
  rotated space. The original-space K is never reconstructed during decode.

Decode for values:
  1. Unpack:      idx = bit_unpack(packed)
  2. Dequantize:  v_hat = scale * idx + zero
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .centroids import (
    decode_indices,
    encode_indices,
    get_boundaries,
    get_centroids,
)
from .hadamard import (
    rotate,
    walsh_hadamard,
)
from .packing import pack, unpack


# ----------------------------------------------------------------------------
# Config / presets — mirror vLLM's TQ_PRESETS and TurboQuantConfig.
# ----------------------------------------------------------------------------

TQ_PRESETS: dict[str, dict] = {
    "turboquant_k8v4": {
        "key_quant_bits": 8,  # FP8 keys (no rotation/MSE)
        "value_quant_bits": 4,
        "norm_correction": False,
    },
    "turboquant_4bit_nc": {
        "key_quant_bits": 4,
        "value_quant_bits": 4,
        "norm_correction": True,
    },
    "turboquant_k3v4_nc": {
        "key_quant_bits": 3,
        "value_quant_bits": 4,
        "norm_correction": True,
    },
    "turboquant_3bit_nc": {
        "key_quant_bits": 3,
        "value_quant_bits": 3,
        "norm_correction": True,
    },
}


@dataclass
class TurboQuantConfig:
    """TurboQuant configuration. Mirrors vLLM's TurboQuantConfig."""

    head_dim: int = 128
    key_quant_bits: int = 3
    value_quant_bits: int = 4
    norm_correction: bool = True

    @property
    def key_fp8(self) -> bool:
        return self.key_quant_bits == 8

    @property
    def n_key_centroids(self) -> int:
        if self.key_fp8:
            return 0
        return 2**self.key_quant_bits

    @property
    def key_packed_bytes(self) -> int:
        """Bytes per (token, head) for keys."""
        if self.key_fp8:
            return self.head_dim  # 1 byte per element for FP8
        from .packing import (
            packed_size_bytes,
        )

        return packed_size_bytes(self.head_dim, self.key_quant_bits) + 2  # +2 for vec_norm fp16

    @property
    def value_packed_bytes(self) -> int:
        """Bytes per (token, head) for values."""
        from .packing import (
            packed_size_bytes,
        )

        return packed_size_bytes(self.head_dim, self.value_quant_bits) + 4  # +scale +zero fp16

    @property
    def total_bytes_per_slot(self) -> int:
        """Total compressed bytes for one (token, head) K+V slot."""
        return self.key_packed_bytes + self.value_packed_bytes

    @property
    def fp16_bytes_per_slot(self) -> int:
        """Uncompressed fp16 bytes for one (token, head) K+V slot."""
        return 2 * self.head_dim * 2  # K + V, fp16 = 2 bytes

    @property
    def compression_ratio(self) -> float:
        """fp16-equivalent size / TurboQuant size."""
        return self.fp16_bytes_per_slot / self.total_bytes_per_slot

    @classmethod
    def from_preset(cls, name: str, head_dim: int = 128) -> "TurboQuantConfig":
        if name not in TQ_PRESETS:
            valid = ", ".join(TQ_PRESETS.keys())
            raise ValueError(f"Unknown preset {name!r}. Valid: {valid}")
        p = TQ_PRESETS[name]
        return cls(
            head_dim=head_dim,
            key_quant_bits=p["key_quant_bits"],
            value_quant_bits=p["value_quant_bits"],
            norm_correction=p["norm_correction"],
        )


# ----------------------------------------------------------------------------
# Encode / decode for keys.
# ----------------------------------------------------------------------------


def encode_keys(
    k: np.ndarray, config: TurboQuantConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize and pack keys.

    Args:
        k: Shape (..., D), float32 or float64. Last dim is head_dim.
        config: TurboQuant configuration.

    Returns:
        packed: Shape (..., packed_bytes), uint8.
        vec_norm: Shape (...), float32. Norms of each row.
    """
    if k.shape[-1] != config.head_dim:
        raise ValueError(f"k last dim {k.shape[-1]} != head_dim {config.head_dim}")
    if config.key_fp8:
        # Defer to a separate FP8 path; not implemented in this reference (we focus
        # on the MSE/Lloyd-Max keys that need the heavy lifting).
        raise NotImplementedError("FP8 key path not implemented in reference")

    # 1. Norm.
    vec_norm = np.linalg.norm(k, axis=-1).astype(np.float32)
    # 2. Normalize, with safe zero handling.
    safe_norm = np.where(vec_norm == 0, 1.0, vec_norm)
    x_hat = (k / safe_norm[..., None]).astype(np.float32)
    # 3. Rotate.
    H = walsh_hadamard(config.head_dim, dtype=np.float32)
    y = rotate(x_hat, H)
    # 4. Quantize via Lloyd-Max boundaries.
    boundaries = get_boundaries(config.head_dim, config.key_quant_bits)
    indices = encode_indices(y, boundaries)
    # 5. Pack.
    packed = pack(indices, config.key_quant_bits)
    return packed, vec_norm


def decode_keys(
    packed: np.ndarray,
    vec_norm: np.ndarray,
    config: TurboQuantConfig,
) -> np.ndarray:
    """Dequantize keys back to the rotated unit-vector space, then denormalize.

    Returns the key in ROTATED space. To get back to original space, apply H
    again (since H is symmetric and self-inverse).

    Args:
        packed: Shape (..., packed_bytes), uint8.
        vec_norm: Shape (...), float32.
        config: TurboQuant configuration.

    Returns:
        k_rot: Shape (..., D), float32, in rotated space.
    """
    if config.key_fp8:
        raise NotImplementedError("FP8 key path not implemented in reference")

    # 1. Unpack.
    indices = unpack(packed, config.head_dim, config.key_quant_bits)
    # 2. Lookup centroids.
    centroids = get_centroids(config.head_dim, config.key_quant_bits)
    y_hat = decode_indices(indices, centroids).astype(np.float32)
    # 3. Optional norm correction.
    if config.norm_correction:
        # The reconstructed vector loses unit norm because of quantization
        # error. Re-normalize the centroid vector to unit length so the
        # reconstructed key has the correct magnitude.
        recon_norm = np.linalg.norm(y_hat, axis=-1)
        safe_recon = np.where(recon_norm == 0, 1.0, recon_norm)
        y_hat = y_hat / safe_recon[..., None]
    # 4. Denormalize.
    k_rot = vec_norm[..., None] * y_hat
    return k_rot


def decode_keys_to_original_space(
    packed: np.ndarray,
    vec_norm: np.ndarray,
    config: TurboQuantConfig,
) -> np.ndarray:
    """Dequantize keys all the way back to the original (non-rotated) space.

    Useful for the bulk-dequant prefill path. NOT used during fused decode —
    decode operates entirely in rotated space.
    """
    k_rot = decode_keys(packed, vec_norm, config)
    H = walsh_hadamard(config.head_dim, dtype=np.float32)
    # H is symmetric and self-inverse, so applying it again recovers the original.
    return rotate(k_rot, H)


# ----------------------------------------------------------------------------
# Encode / decode for values.
# ----------------------------------------------------------------------------


def encode_values(
    v: np.ndarray, config: TurboQuantConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniform asymmetric quantization for values.

    Args:
        v: Shape (..., D), float32 or float64.
        config: TurboQuant configuration.

    Returns:
        packed: Shape (..., packed_bytes), uint8.
        v_scale: Shape (...), float32. Scale per row.
        v_zero:  Shape (...), float32. Zero point (= min) per row.
    """
    if v.shape[-1] != config.head_dim:
        raise ValueError(f"v last dim {v.shape[-1]} != head_dim {config.head_dim}")
    n_levels = 2**config.value_quant_bits
    v_min = v.min(axis=-1).astype(np.float32)
    v_max = v.max(axis=-1).astype(np.float32)
    v_scale = ((v_max - v_min) / (n_levels - 1)).astype(np.float32)
    # Avoid divide-by-zero for constant rows.
    safe_scale = np.where(v_scale == 0, 1.0, v_scale)
    indices = np.round((v - v_min[..., None]) / safe_scale[..., None]).astype(np.int32)
    indices = np.clip(indices, 0, n_levels - 1).astype(np.uint8)
    packed = pack(indices, config.value_quant_bits)
    return packed, v_scale, v_min


def decode_values(
    packed: np.ndarray,
    v_scale: np.ndarray,
    v_zero: np.ndarray,
    config: TurboQuantConfig,
) -> np.ndarray:
    """Dequantize values."""
    indices = unpack(packed, config.head_dim, config.value_quant_bits)
    return (v_scale[..., None] * indices.astype(np.float32) + v_zero[..., None]).astype(
        np.float32
    )


# ----------------------------------------------------------------------------
# Attention scoring in rotated space (the critical decode path).
# ----------------------------------------------------------------------------


def score_in_rotated_space(
    q: np.ndarray,
    packed_k: np.ndarray,
    vec_norm: np.ndarray,
    config: TurboQuantConfig,
    H: np.ndarray | None = None,
) -> np.ndarray:
    """Compute Q . K scores using TurboQuant-encoded K, without dequantizing
    K back to original space.

    This is the kernel-level operation we have to fuse on GPU for performance.
    The math:
        score = q . k
              = q . (n * H * y_hat)         # k = norm * H * y_hat (rotated unit vec)
              = n * (q @ H) . y_hat         # H is symmetric/orthogonal
              = n * (q_rot . y_hat)
    So we rotate Q ONCE per layer per step, then for each token in the cache
    do a lookup + dot-product.

    Args:
        q: Shape (..., D), the query in original space.
        packed_k: Shape (S, packed_bytes), uint8 (S = seq positions).
        vec_norm: Shape (S,), float32.
        config: TurboQuant configuration.
        H: Optional precomputed Hadamard.

    Returns:
        scores: Shape (..., S), float32.
    """
    if H is None:
        H = walsh_hadamard(config.head_dim, dtype=np.float32)
    q_rot = rotate(q.astype(np.float32), H)  # (..., D)

    # Decode k to rotated space.
    indices = unpack(packed_k, config.head_dim, config.key_quant_bits)
    centroids = get_centroids(config.head_dim, config.key_quant_bits)
    y_hat = centroids[indices].astype(np.float32)  # (S, D)
    if config.norm_correction:
        recon_norm = np.linalg.norm(y_hat, axis=-1)
        safe = np.where(recon_norm == 0, 1.0, recon_norm)
        y_hat = y_hat / safe[..., None]
    # k_rot = n * y_hat;  score = q_rot . k_rot = n * q_rot . y_hat
    # einsum over the last axis.
    scores = np.einsum("...d,sd->...s", q_rot, y_hat) * vec_norm
    return scores

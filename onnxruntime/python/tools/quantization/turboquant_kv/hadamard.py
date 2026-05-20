# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Walsh-Hadamard rotation for TurboQuant.

We use the Sylvester construction (recursive 2x2 expansion), normalized by
1/sqrt(d). This matrix is:
  - Symmetric: H = H^T
  - Self-inverse (after normalization): H @ H = I
  - Orthogonal: H @ H^T = I

Properties for TurboQuant:
  - Rotating any unit vector by H gives a vector whose coordinates are
    approximately N(0, 1/d) for d >= 64. This is the distribution Lloyd-Max
    is optimized for.
  - Because H is symmetric, the inverse rotation is just another multiplication
    by H. This means we never need to store H^T separately.
  - Because attention scores are inner products and inner products are
    invariant under orthogonal transforms, attention can be computed entirely
    in the rotated space:
      (q @ H) . (k_hat @ H) * ||k|| == q . k

WARNING: row-major / column-major trap.
  The matrix returned here is row-major C-order. In any framework that
  interprets contiguous storage as column-major (Fortran, some BLAS APIs)
  or that implicitly transposes on tensor wrap, you will compute H^T x
  instead of H x. Because H is symmetric (H = H^T) this happens to give the
  same numerical result for our use case, BUT documenting the property
  matters because it doesn't generalize to randomized rotations.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=8)
def walsh_hadamard(d: int, dtype: type = np.float32) -> np.ndarray:
    """Build the d x d normalized Walsh-Hadamard matrix via Sylvester recursion.

    H_1 = [[1]]
    H_{2n} = [[H_n,  H_n],
              [H_n, -H_n]]

    Then divide by sqrt(d) so that H @ H^T = I.

    Args:
        d: Dimension (must be a power of 2).
        dtype: Output dtype.

    Returns:
        H: Shape (d, d), symmetric, self-inverse after normalization.
    """
    if d <= 0 or (d & (d - 1)) != 0:
        raise ValueError(f"head_dim must be a positive power of 2, got {d}")

    H = np.array([[1.0]], dtype=np.float64)
    while H.shape[0] < d:
        H = np.block([[H, H], [H, -H]])
    H = H / math.sqrt(d)
    return H.astype(dtype)


def rotate(x: np.ndarray, H: np.ndarray | None = None) -> np.ndarray:
    """Apply Walsh-Hadamard rotation to the last axis of x.

    Args:
        x: Shape (..., D).
        H: Optional precomputed Hadamard. If None, built from x.shape[-1].

    Returns:
        x @ H of the same shape and dtype as x.
    """
    d = x.shape[-1]
    if H is None:
        H = walsh_hadamard(d, dtype=x.dtype)
    elif H.shape != (d, d):
        raise ValueError(f"H shape {H.shape} doesn't match D={d}")
    return x @ H


def fast_walsh_hadamard_transform(x: np.ndarray) -> np.ndarray:
    """In-place fast Walsh-Hadamard transform (FWHT), butterfly form.

    Computes H @ x in O(d log d) time without materializing H. Used for
    benchmarking / future kernels — for the reference impl we use the
    explicit matmul above which is O(d^2) but more obviously correct.

    Args:
        x: Shape (..., D), D must be a power of 2.

    Returns:
        Rotated array, same shape and dtype.
    """
    x = x.copy()
    d = x.shape[-1]
    if d <= 0 or (d & (d - 1)) != 0:
        raise ValueError(f"D must be a positive power of 2, got {d}")
    h = 1
    while h < d:
        for i in range(0, d, h * 2):
            for j in range(i, i + h):
                a = x[..., j].copy()
                b = x[..., j + h].copy()
                x[..., j] = a + b
                x[..., j + h] = a - b
        h *= 2
    return x / math.sqrt(d)

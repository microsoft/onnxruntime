# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Lloyd-Max optimal scalar quantizer for TurboQuant.

After rotating a d-dimensional unit vector by an orthogonal Hadamard matrix,
each coordinate is approximately N(0, 1/d) for d >= 64 (concentration of
measure). We solve the Lloyd-Max conditions to find optimal centroids that
minimize the expected mean squared error of scalar quantization under that
distribution.

Algorithm (Lloyd 1957 / Max 1960):
  Iterate two steps until convergence:
    1. Boundaries: b_i = (c_i + c_{i+1}) / 2 (Voronoi midpoints).
    2. Centroids:  c_i = E[X | b_{i-1} < X <= b_i]
                       = integral(x * f(x), b_{i-1}, b_i) / integral(f(x), b_{i-1}, b_i)

Ported from vLLM's turboquant/centroids.py (which was in turn based on
turboquant-pytorch/lloyd_max.py by Zandieh et al.). Adapted to pure NumPy
(no torch / scipy dependency).
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np


def _gaussian_pdf(x: float, sigma2: float) -> float:
    """N(0, sigma2) probability density at x."""
    return (1.0 / math.sqrt(2 * math.pi * sigma2)) * math.exp(-x * x / (2 * sigma2))


def _trapz(f, a: float, b: float, n: int = 200) -> float:
    """Trapezoidal numerical integration of f over [a, b] with n subintervals."""
    h = (b - a) / n
    result = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        result += f(a + i * h)
    return result * h


def solve_lloyd_max(
    d: int,
    bits: int,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve Lloyd-Max optimal quantizer for N(0, 1/d) distribution.

    Args:
        d: Vector dimension (head_dim). Determines variance = 1/d.
        bits: Number of quantization bits (3 or 4 typical).
        max_iter: Maximum Lloyd-Max iterations.
        tol: Convergence tolerance on max centroid drift.

    Returns:
        centroids: Sorted array of 2^bits float32 optimal centroids.
        boundaries: Sorted array of 2^bits - 1 float32 decision boundaries.

    Properties of the solution:
        - Symmetric around 0 (because N(0, 1/d) is symmetric).
        - For bits=3, gives 8 levels; bits=4 gives 16 levels.
        - The codebook is shared across ALL coordinates of ALL keys. There
          is no per-head, per-channel, or per-token codebook — this is the
          key efficiency win of TurboQuant.
    """
    n_levels = 2**bits
    sigma2 = 1.0 / d
    sigma = math.sqrt(sigma2)

    def pdf(x: float) -> float:
        return _gaussian_pdf(x, sigma2)

    # Initial centroids: evenly spaced in [-3.5σ, 3.5σ].
    lo, hi = -3.5 * sigma, 3.5 * sigma
    centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

    for _ in range(max_iter):
        boundaries = [
            (centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)
        ]
        # Open intervals at the ends (push to ±3*hi for ~zero PDF mass).
        edges = [lo * 3.0] + boundaries + [hi * 3.0]
        new_centroids = []
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]
            num = _trapz(lambda x: x * pdf(x), a, b)
            den = _trapz(pdf, a, b)
            new_centroids.append(num / den if den > 1e-15 else centroids[i])

        drift = max(abs(new_centroids[i] - centroids[i]) for i in range(n_levels))
        centroids = new_centroids
        if drift < tol:
            break

    boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
    return (
        np.asarray(centroids, dtype=np.float32),
        np.asarray(boundaries, dtype=np.float32),
    )


@lru_cache(maxsize=32)
def get_centroids(d: int, bits: int) -> np.ndarray:
    """Get cached Lloyd-Max centroids for (d, bits)."""
    centroids, _ = solve_lloyd_max(d, bits)
    return centroids


@lru_cache(maxsize=32)
def get_boundaries(d: int, bits: int) -> np.ndarray:
    """Get cached Lloyd-Max decision boundaries for (d, bits)."""
    _, boundaries = solve_lloyd_max(d, bits)
    return boundaries


def encode_indices(y: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """Map values to centroid indices via decision boundaries (binary search).

    Args:
        y: Values to quantize, shape (..., D), any float dtype.
        boundaries: Sorted decision boundaries, shape (n_levels - 1,).

    Returns:
        Indices in [0, n_levels), uint8.
    """
    # np.searchsorted returns the index where each y would be inserted to keep
    # the boundaries sorted. With side='right', we get the inclusive bin index.
    indices = np.searchsorted(boundaries, y, side="right")
    return indices.astype(np.uint8)


def decode_indices(indices: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Decode centroid indices back to float values."""
    return centroids[indices]

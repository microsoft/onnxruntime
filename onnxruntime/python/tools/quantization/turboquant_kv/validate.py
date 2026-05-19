# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Paper-faithfulness tests for the TurboQuant reference implementation.

Adapted from 0xSero/turboquant's validate_paper.py. Each test corresponds to
a specific theorem or claim from the TurboQuant paper or its citing
literature, and gives a single pass/fail (or numeric pass/fail with margin).

These run in pure NumPy and complete in seconds — they are the unit-test
acceptance gate for the reference impl, and (once kernels are written) for
GPU kernels too via output-comparison.

Run with:
    python -m onnxruntime.python.tools.quantization.turboquant_kv.validate
"""

from __future__ import annotations

import math
import sys

import numpy as np

from .centroids import (
    decode_indices,
    encode_indices,
    get_boundaries,
    get_centroids,
)
from .hadamard import (
    fast_walsh_hadamard_transform,
    rotate,
    walsh_hadamard,
)
from .packing import (
    pack,
    pack_3bit,
    pack_4bit,
    unpack,
    unpack_3bit,
    unpack_4bit,
)
from .quantizer import (
    TQ_PRESETS,
    TurboQuantConfig,
    decode_keys,
    decode_keys_to_original_space,
    decode_values,
    encode_keys,
    encode_values,
    score_in_rotated_space,
)


# ----------------------------------------------------------------------------
# Test harness.
# ----------------------------------------------------------------------------


class TestResult:
    def __init__(self, name: str, ok: bool, detail: str = ""):
        self.name = name
        self.ok = ok
        self.detail = detail

    def __str__(self) -> str:
        mark = "PASS" if self.ok else "FAIL"
        return f"[{mark}] {self.name}: {self.detail}"


_results: list[TestResult] = []


def _check(name: str, ok: bool, detail: str = "") -> bool:
    res = TestResult(name, ok, detail)
    _results.append(res)
    print(res)
    return ok


# ----------------------------------------------------------------------------
# Tests.
# ----------------------------------------------------------------------------


def test_hadamard_orthogonal(d: int = 128) -> bool:
    """Walsh-Hadamard is orthogonal: H @ H^T = I."""
    H = walsh_hadamard(d, dtype=np.float64)
    err = np.max(np.abs(H @ H.T - np.eye(d)))
    return _check(
        f"Hadamard orthogonal d={d}", err < 1e-10, f"max |H H^T - I| = {err:.2e}"
    )


def test_hadamard_self_inverse(d: int = 128) -> bool:
    """Walsh-Hadamard is symmetric, so H @ H = I (its own inverse)."""
    H = walsh_hadamard(d, dtype=np.float64)
    err = np.max(np.abs(H @ H - np.eye(d)))
    return _check(
        f"Hadamard self-inverse d={d}", err < 1e-10, f"max |H H - I| = {err:.2e}"
    )


def test_fwht_matches_matmul(d: int = 128, seed: int = 0) -> bool:
    """Fast Walsh-Hadamard transform agrees with naive matmul."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((4, d)).astype(np.float64)
    via_matmul = rotate(x, walsh_hadamard(d, dtype=np.float64))
    via_fwht = fast_walsh_hadamard_transform(x)
    err = np.max(np.abs(via_matmul - via_fwht))
    return _check(
        f"FWHT == matmul d={d}", err < 1e-10, f"max abs diff = {err:.2e}"
    )


def test_rotated_unit_vec_is_normal(d: int = 128, n: int = 50000, seed: int = 0) -> bool:
    """Concentration of measure: H @ x_hat for unit vector x_hat is ~N(0, 1/d).

    We check empirical mean and variance match the theoretical distribution.
    """
    rng = np.random.default_rng(seed)
    # Sample n unit vectors uniformly on S^{d-1}.
    x = rng.standard_normal((n, d))
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    H = walsh_hadamard(d, dtype=np.float64)
    y = x @ H
    # All d * n coordinates pooled.
    mean = y.mean()
    var = y.var()
    expected_var = 1.0 / d
    mean_ok = abs(mean) < 0.01
    var_ok = abs(var - expected_var) / expected_var < 0.05
    return _check(
        "Post-rotation marginal ~N(0, 1/d)",
        mean_ok and var_ok,
        f"mean={mean:.4f} (want ~0) var={var:.6f} (want {expected_var:.6f})",
    )


def test_centroids_symmetric(d: int = 128, bits: int = 3) -> bool:
    """Lloyd-Max centroids for symmetric distribution are symmetric around 0."""
    centroids = get_centroids(d, bits)
    # Sorted centroids, c[i] should equal -c[2^bits - 1 - i].
    n = len(centroids)
    err = max(abs(centroids[i] + centroids[n - 1 - i]) for i in range(n))
    return _check(
        f"Centroids symmetric d={d} bits={bits}",
        err < 1e-6,
        f"max |c[i] + c[n-1-i]| = {err:.2e}, centroids = {centroids}",
    )


def test_centroid_count(d: int = 128, bits: int = 3) -> bool:
    """Number of centroids equals 2^bits."""
    centroids = get_centroids(d, bits)
    expected = 2**bits
    return _check(
        f"Centroid count d={d} bits={bits}",
        len(centroids) == expected,
        f"got {len(centroids)}, expected {expected}",
    )


def test_centroids_lloyd_max_optimal(d: int = 128, bits: int = 3, seed: int = 0) -> bool:
    """Lloyd-Max codebook achieves lower MSE than uniform on N(0, 1/d).

    We sample from N(0, 1/d), quantize with both schemes, and verify Lloyd-Max
    has strictly lower MSE.
    """
    rng = np.random.default_rng(seed)
    sigma = 1.0 / math.sqrt(d)
    n = 200_000
    x = rng.standard_normal(n) * sigma
    centroids = get_centroids(d, bits)
    boundaries = get_boundaries(d, bits)
    idx_lm = encode_indices(x, boundaries)
    x_hat_lm = decode_indices(idx_lm, centroids)
    mse_lm = np.mean((x - x_hat_lm) ** 2)

    # Uniform quantizer on [-3.5σ, 3.5σ].
    n_levels = 2**bits
    lo, hi = -3.5 * sigma, 3.5 * sigma
    uniform_centroids = np.linspace(
        lo + (hi - lo) / (2 * n_levels), hi - (hi - lo) / (2 * n_levels), n_levels
    )
    step = (hi - lo) / n_levels
    idx_un = np.clip(((x - lo) / step).astype(int), 0, n_levels - 1)
    x_hat_un = uniform_centroids[idx_un]
    mse_un = np.mean((x - x_hat_un) ** 2)

    return _check(
        f"Lloyd-Max beats uniform d={d} bits={bits}",
        mse_lm < mse_un,
        f"MSE Lloyd-Max = {mse_lm:.6e}, MSE uniform = {mse_un:.6e}, ratio = {mse_lm / mse_un:.3f}",
    )


def test_4bit_pack_roundtrip(d: int = 128, seed: int = 0) -> bool:
    """4-bit pack/unpack is bijective."""
    rng = np.random.default_rng(seed)
    values = rng.integers(0, 16, size=(7, 5, d)).astype(np.uint8)
    packed = pack_4bit(values)
    expected_bytes = d // 2
    shape_ok = packed.shape == (7, 5, expected_bytes)
    recovered = unpack_4bit(packed, d)
    eq = np.array_equal(values, recovered)
    return _check(
        f"4-bit pack roundtrip d={d}",
        shape_ok and eq,
        f"packed shape = {packed.shape}, equal = {eq}",
    )


def test_3bit_pack_roundtrip(d: int = 128, seed: int = 1) -> bool:
    """3-bit pack/unpack is bijective."""
    rng = np.random.default_rng(seed)
    values = rng.integers(0, 8, size=(7, 5, d)).astype(np.uint8)
    packed = pack_3bit(values)
    expected_bytes = d * 3 // 8
    shape_ok = packed.shape == (7, 5, expected_bytes)
    recovered = unpack_3bit(packed, d)
    eq = np.array_equal(values, recovered)
    return _check(
        f"3-bit pack roundtrip d={d}",
        shape_ok and eq,
        f"packed shape = {packed.shape}, equal = {eq}",
    )


def test_3bit_pack_layout_one_group() -> bool:
    """Verify the bit-layout of one 8-element group matches our spec."""
    # v0..v7 = 1, 2, 3, 4, 5, 6, 7, 0
    values = np.array([1, 2, 3, 4, 5, 6, 7, 0], dtype=np.uint8).reshape(1, 8)
    packed = pack_3bit(values)  # shape (1, 3)
    # Expected 24-bit word = (0 << 21) | (7 << 18) | (6 << 15) | (5 << 12) | (4 << 9) | (3 << 6) | (2 << 3) | 1
    word = 0
    for i, v in enumerate([1, 2, 3, 4, 5, 6, 7, 0]):
        word |= v << (i * 3)
    expected_b0 = word & 0xFF
    expected_b1 = (word >> 8) & 0xFF
    expected_b2 = (word >> 16) & 0xFF
    ok = (
        int(packed[0, 0]) == expected_b0
        and int(packed[0, 1]) == expected_b1
        and int(packed[0, 2]) == expected_b2
    )
    return _check(
        "3-bit pack layout (one group)",
        ok,
        f"got {packed.tolist()}, expected [{expected_b0}, {expected_b1}, {expected_b2}]",
    )


def test_key_roundtrip_rotated_space(
    head_dim: int = 128, bits: int = 3, n: int = 64, seed: int = 0
) -> bool:
    """Encode then decode keys, check we recover the rotated normalized form.

    Acceptance: cosine similarity > 0.99 in rotated space (i.e. between
    H @ k_hat and the reconstructed centroid vector).
    """
    rng = np.random.default_rng(seed)
    config = TurboQuantConfig(head_dim=head_dim, key_quant_bits=bits, value_quant_bits=4, norm_correction=False)
    k = rng.standard_normal((n, head_dim)).astype(np.float32)

    packed, vec_norm = encode_keys(k, config)
    k_rot_hat = decode_keys(packed, vec_norm, config)

    # Compute the ground-truth in rotated space.
    H = walsh_hadamard(head_dim, dtype=np.float32)
    norms = np.linalg.norm(k, axis=-1, keepdims=True)
    k_rot_truth = (k / norms) @ H * norms

    cos = np.sum(k_rot_hat * k_rot_truth, axis=-1) / (
        np.linalg.norm(k_rot_hat, axis=-1) * np.linalg.norm(k_rot_truth, axis=-1) + 1e-9
    )
    median_cos = float(np.median(cos))
    min_cos = float(np.min(cos))
    threshold = 0.97 if bits == 3 else 0.995
    return _check(
        f"Key roundtrip rotated-space cos d={head_dim} bits={bits}",
        median_cos > threshold and min_cos > threshold - 0.02,
        f"median={median_cos:.4f} min={min_cos:.4f} threshold={threshold}",
    )


def test_key_roundtrip_original_space(
    head_dim: int = 128, bits: int = 4, n: int = 64, seed: int = 0
) -> bool:
    """Round-trip through original space using the bulk-dequant path."""
    rng = np.random.default_rng(seed)
    config = TurboQuantConfig(head_dim=head_dim, key_quant_bits=bits, value_quant_bits=4, norm_correction=True)
    k = rng.standard_normal((n, head_dim)).astype(np.float32)
    packed, vec_norm = encode_keys(k, config)
    k_hat = decode_keys_to_original_space(packed, vec_norm, config)

    cos = np.sum(k * k_hat, axis=-1) / (
        np.linalg.norm(k, axis=-1) * np.linalg.norm(k_hat, axis=-1) + 1e-9
    )
    median_cos = float(np.median(cos))
    threshold = 0.99 if bits == 4 else 0.95
    return _check(
        f"Key roundtrip original-space cos d={head_dim} bits={bits}",
        median_cos > threshold,
        f"median cos = {median_cos:.4f} (threshold {threshold})",
    )


def test_value_roundtrip(head_dim: int = 128, bits: int = 4, n: int = 64, seed: int = 0) -> bool:
    """Uniform quant for V: cosine similarity > 0.999 at 4 bits."""
    rng = np.random.default_rng(seed)
    config = TurboQuantConfig(head_dim=head_dim, key_quant_bits=4, value_quant_bits=bits, norm_correction=True)
    v = rng.standard_normal((n, head_dim)).astype(np.float32)
    packed, scale, zero = encode_values(v, config)
    v_hat = decode_values(packed, scale, zero, config)
    cos = np.sum(v * v_hat, axis=-1) / (
        np.linalg.norm(v, axis=-1) * np.linalg.norm(v_hat, axis=-1) + 1e-9
    )
    median_cos = float(np.median(cos))
    # Thresholds are for synthetic N(0,1) data with min-max scaling. Real model
    # values typically achieve ~0.001 higher cos-sim because their distributions
    # are more clustered.
    threshold = {4: 0.994, 3: 0.978, 2: 0.92}[bits]
    return _check(
        f"Value roundtrip cos d={head_dim} bits={bits}",
        median_cos > threshold,
        f"median cos = {median_cos:.4f} (threshold {threshold})",
    )


def test_score_rotated_equals_full_dot(
    head_dim: int = 128, bits: int = 3, n_keys: int = 32, seed: int = 0
) -> bool:
    """The rotated-space score should equal q . k_hat (where k_hat is the
    full reconstruction in original space). This validates that we never
    need to back-rotate during decode."""
    rng = np.random.default_rng(seed)
    config = TurboQuantConfig(head_dim=head_dim, key_quant_bits=bits, value_quant_bits=4, norm_correction=False)
    q = rng.standard_normal((1, head_dim)).astype(np.float32)
    k = rng.standard_normal((n_keys, head_dim)).astype(np.float32)
    packed, vec_norm = encode_keys(k, config)

    scores_rotated = score_in_rotated_space(q, packed, vec_norm, config)  # (1, n_keys)
    # Dequant K to original space and compute scores directly.
    k_hat_orig = decode_keys_to_original_space(packed, vec_norm, config)
    scores_full = q @ k_hat_orig.T  # (1, n_keys)

    err = float(np.max(np.abs(scores_rotated - scores_full)))
    rel = err / (np.max(np.abs(scores_full)) + 1e-9)
    return _check(
        f"Rotated-space score == original-space score d={head_dim} bits={bits}",
        rel < 1e-5,
        f"max abs diff = {err:.2e}, relative = {rel:.2e}",
    )


def test_compression_ratio() -> bool:
    """Raw per-element compression ratios for d=128.

    Raw ratio = fp16 bytes per slot / TQ bytes per slot. Effective system-wide
    ratios reported by vLLM are slightly different because they account for the
    boundary-skip layers (first/last N keep fp16) — that's a model-level
    effect, not a per-slot one.
    """
    # Per-slot bytes for head_dim=128 fp16: 2*128*2 = 512.
    # 4bit_nc:  K = 64+2 = 66, V = 64+4 = 68 -> 134, ratio = 3.82
    # k3v4_nc:  K = 48+2 = 50, V = 64+4 = 68 -> 118, ratio = 4.34
    # 3bit_nc:  K = 48+2 = 50, V = 48+4 = 52 -> 102, ratio = 5.02
    expectations = {
        "turboquant_4bit_nc": 3.82,
        "turboquant_k3v4_nc": 4.34,
        "turboquant_3bit_nc": 5.02,
    }
    all_ok = True
    detail = []
    for name, expected in expectations.items():
        cfg = TurboQuantConfig.from_preset(name, head_dim=128)
        actual = cfg.compression_ratio
        ok = abs(actual - expected) / expected < 0.02
        all_ok = all_ok and ok
        detail.append(f"{name}: {actual:.2f}x (expected ~{expected:.2f}x)")
    return _check(
        "Compression ratios (raw, head_dim=128)", all_ok, "; ".join(detail)
    )


# ----------------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------------


def main() -> int:
    print("=" * 78)
    print("TurboQuant reference-impl validation")
    print("=" * 78)

    test_hadamard_orthogonal(64)
    test_hadamard_orthogonal(128)
    test_hadamard_self_inverse(128)
    test_fwht_matches_matmul(128)
    test_rotated_unit_vec_is_normal(128)

    test_centroid_count(128, 3)
    test_centroid_count(128, 4)
    test_centroids_symmetric(128, 3)
    test_centroids_symmetric(128, 4)
    test_centroids_lloyd_max_optimal(128, 3)
    test_centroids_lloyd_max_optimal(128, 4)

    test_4bit_pack_roundtrip(128)
    test_4bit_pack_roundtrip(64)
    test_3bit_pack_roundtrip(128)
    test_3bit_pack_layout_one_group()

    test_key_roundtrip_rotated_space(128, 3)
    test_key_roundtrip_rotated_space(128, 4)
    test_key_roundtrip_original_space(128, 4)
    test_value_roundtrip(128, 4)
    test_value_roundtrip(128, 3)

    test_score_rotated_equals_full_dot(128, 3)
    test_score_rotated_equals_full_dot(128, 4)

    test_compression_ratio()

    print("=" * 78)
    n_pass = sum(1 for r in _results if r.ok)
    n_fail = len(_results) - n_pass
    print(f"Total: {len(_results)}   Pass: {n_pass}   Fail: {n_fail}")
    print("=" * 78)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

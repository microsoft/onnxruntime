# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""TurboQuant benchmark suite — compares against fp16 KV-cache baseline.

Measures three things:
  1. Memory: bytes per KV-slot (raw + system-level for popular models).
  2. Accuracy: cosine similarity of attention scores vs fp16 baseline.
  3. Throughput: encode / decode operations per second.

Run with:
    python -m onnxruntime.quantization.turboquant_kv.benchmark
or while developing:
    cd onnxruntime/python/tools/quantization
    python -m turboquant_kv.benchmark
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np

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
from .hadamard import walsh_hadamard


# ----------------------------------------------------------------------------
# Model presets (head_dim, n_heads, n_kv_heads, n_layers, max_seq).
# These are real model dimensions used to compute realistic memory savings.
# ----------------------------------------------------------------------------


@dataclass
class ModelSpec:
    name: str
    n_layers: int
    n_kv_heads: int
    head_dim: int

    def fp16_kv_bytes(self, batch: int, seq: int) -> int:
        """fp16 KV cache size in bytes."""
        return 2 * batch * self.n_layers * self.n_kv_heads * seq * self.head_dim * 2

    def tq_kv_bytes(self, batch: int, seq: int, cfg: TurboQuantConfig) -> int:
        """TurboQuant KV cache size in bytes (raw, no boundary skip)."""
        per_slot = cfg.total_bytes_per_slot
        return batch * self.n_layers * self.n_kv_heads * seq * per_slot

    def tq_kv_bytes_with_boundary(
        self, batch: int, seq: int, cfg: TurboQuantConfig, boundary_n: int = 2
    ) -> int:
        """TurboQuant KV cache size with first/last `boundary_n` layers in fp16."""
        boundary_layers = min(2 * boundary_n, self.n_layers)
        compressed_layers = self.n_layers - boundary_layers
        compressed = batch * compressed_layers * self.n_kv_heads * seq * cfg.total_bytes_per_slot
        boundary = 2 * batch * boundary_layers * self.n_kv_heads * seq * self.head_dim * 2
        return compressed + boundary


MODELS: list[ModelSpec] = [
    # (name, layers, kv_heads, head_dim) for popular open-weight LLMs.
    ModelSpec("Llama-3.1-8B", n_layers=32, n_kv_heads=8, head_dim=128),
    ModelSpec("Llama-3.1-70B", n_layers=80, n_kv_heads=8, head_dim=128),
    ModelSpec("Qwen2.5-7B", n_layers=28, n_kv_heads=4, head_dim=128),
    ModelSpec("Qwen2.5-72B", n_layers=80, n_kv_heads=8, head_dim=128),
    ModelSpec("Mistral-7B", n_layers=32, n_kv_heads=8, head_dim=128),
    ModelSpec("Phi-3.5-mini", n_layers=32, n_kv_heads=32, head_dim=96),
    ModelSpec("Gemma-2-9B", n_layers=42, n_kv_heads=8, head_dim=256),
]


PRESETS_TO_TEST = ["turboquant_4bit_nc", "turboquant_k3v4_nc", "turboquant_3bit_nc"]


# ----------------------------------------------------------------------------
# Helpers.
# ----------------------------------------------------------------------------


def fmt_bytes(n: int) -> str:
    if n >= 1 << 30:
        return f"{n / (1 << 30):.2f} GB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.2f} MB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.2f} KB"
    return f"{n} B"


def cosine(a: np.ndarray, b: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.sum(a * b, axis=axis) / (
        np.linalg.norm(a, axis=axis) * np.linalg.norm(b, axis=axis) + 1e-9
    )


# ----------------------------------------------------------------------------
# Memory benchmark.
# ----------------------------------------------------------------------------


def benchmark_memory(seq_lens: list[int], head_dim: int = 128) -> str:
    """Print memory savings for each model + preset across context lengths."""
    out = []
    out.append("\n## 1. Memory: KV cache bytes (batch=1)\n")
    out.append(
        f"{'Model':<18} {'Preset':<24} "
        + " ".join(f"{f'seq={s}':>10}" for s in seq_lens)
    )
    out.append("-" * (18 + 24 + 11 * len(seq_lens)))
    for model in MODELS:
        if model.head_dim != head_dim:
            continue
        # fp16 baseline
        fp16_row = f"{model.name:<18} {'fp16 (baseline)':<24} " + " ".join(
            f"{fmt_bytes(model.fp16_kv_bytes(1, s)):>10}" for s in seq_lens
        )
        out.append(fp16_row)
        for preset in PRESETS_TO_TEST:
            cfg = TurboQuantConfig.from_preset(preset, head_dim=model.head_dim)
            row = f"{'':<18} {preset:<24} " + " ".join(
                f"{fmt_bytes(model.tq_kv_bytes(1, s, cfg)):>10}" for s in seq_lens
            )
            out.append(row)
        # Compression ratio summary line.
        ratios = [
            TurboQuantConfig.from_preset(p, head_dim=model.head_dim).compression_ratio
            for p in PRESETS_TO_TEST
        ]
        out.append(
            f"{'':<18} {'compression x':<24} "
            + " ".join(f"{r:>9.2f}x" for r in ratios)
        )
    out.append("")
    return "\n".join(out)


def benchmark_memory_per_slot() -> str:
    """Per-slot byte budget for each preset at head_dim 64, 96, 128, 256."""
    out = []
    out.append("\n## 1b. Per-slot bytes (one (token, head) K+V slot)\n")
    out.append(f"{'head_dim':>9} {'preset':<22} {'fp16':>8} {'TQ':>8} {'ratio':>8}")
    out.append("-" * 60)
    for d in [64, 96, 128, 256]:
        for preset in PRESETS_TO_TEST:
            cfg = TurboQuantConfig.from_preset(preset, head_dim=d)
            fp16_b = cfg.fp16_bytes_per_slot
            tq_b = cfg.total_bytes_per_slot
            out.append(
                f"{d:>9} {preset:<22} {fp16_b:>8} {tq_b:>8} {fp16_b / tq_b:>7.2f}x"
            )
        out.append("")
    return "\n".join(out)


# ----------------------------------------------------------------------------
# Accuracy benchmark.
# ----------------------------------------------------------------------------


def benchmark_accuracy(
    head_dim: int = 128, n_keys: int = 1024, n_queries: int = 128, seed: int = 0
) -> str:
    """Cosine similarity of attention scores vs fp16 baseline.

    For each preset we:
      1. Generate random Q, K, V (Gaussian).
      2. Compute fp16 attention scores: softmax(Q K^T / sqrt(d))
      3. Compute TQ attention scores using the rotated-space path.
      4. Compute V output: softmax(...) @ V (uniform-quant V)
      5. Report cosine sim of (a) scores, (b) softmax outputs, (c) attention output.
    """
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((n_queries, head_dim)).astype(np.float32)
    K = rng.standard_normal((n_keys, head_dim)).astype(np.float32)
    V = rng.standard_normal((n_keys, head_dim)).astype(np.float32)
    scale = 1.0 / np.sqrt(head_dim)
    scores_fp16 = (Q @ K.T) * scale  # (Nq, Nk)
    weights_fp16 = _softmax(scores_fp16)
    out_fp16 = weights_fp16 @ V  # (Nq, D)

    out_lines = []
    out_lines.append("\n## 2. Accuracy vs fp16 baseline (synthetic Gaussian)\n")
    out_lines.append(f"head_dim = {head_dim}, n_queries = {n_queries}, n_keys = {n_keys}")
    out_lines.append(
        f"{'preset':<22} {'score cos':>10} {'softmax cos':>12} {'output cos':>12}"
    )
    out_lines.append("-" * 60)

    for preset in PRESETS_TO_TEST:
        cfg = TurboQuantConfig.from_preset(preset, head_dim=head_dim)

        # Encode K and V.
        packed_k, vec_norm = encode_keys(K, cfg)
        packed_v, v_scale, v_zero = encode_values(V, cfg)

        # Compute scores in rotated space.
        scores_tq = score_in_rotated_space(Q, packed_k, vec_norm, cfg) * scale  # (Nq, Nk)
        weights_tq = _softmax(scores_tq)
        V_hat = decode_values(packed_v, v_scale, v_zero, cfg)  # (Nk, D)
        out_tq = weights_tq @ V_hat

        score_cos = float(np.median(cosine(scores_fp16, scores_tq)))
        softmax_cos = float(np.median(cosine(weights_fp16, weights_tq)))
        output_cos = float(np.median(cosine(out_fp16, out_tq)))
        out_lines.append(
            f"{preset:<22} {score_cos:>10.5f} {softmax_cos:>12.5f} {output_cos:>12.5f}"
        )

    return "\n".join(out_lines)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


# ----------------------------------------------------------------------------
# Throughput benchmark.
# ----------------------------------------------------------------------------


def benchmark_throughput(head_dim: int = 128, n_tokens: int = 16384) -> str:
    """Encode + decode throughput in MTokens/s on CPU.

    Numbers are NumPy-on-CPU lower bounds. The CUDA / WebGPU kernels will be
    100-1000x faster at decode (rotated-space dot product is parallelized
    across heads + tokens).
    """
    rng = np.random.default_rng(0)
    K = rng.standard_normal((n_tokens, head_dim)).astype(np.float32)
    V = rng.standard_normal((n_tokens, head_dim)).astype(np.float32)
    Q = rng.standard_normal((1, head_dim)).astype(np.float32)
    H = walsh_hadamard(head_dim, dtype=np.float32)

    out_lines = []
    out_lines.append("\n## 3. CPU throughput (NumPy reference, lower bound)\n")
    out_lines.append(
        f"head_dim = {head_dim}, n_tokens = {n_tokens}"
    )
    out_lines.append(
        f"{'preset':<22} {'enc K MT/s':>12} {'enc V MT/s':>12} {'score MT/s':>12} {'dec V MT/s':>12}"
    )
    out_lines.append("-" * 75)

    for preset in PRESETS_TO_TEST:
        cfg = TurboQuantConfig.from_preset(preset, head_dim=head_dim)

        # Time K encode (rotation + Lloyd-Max + pack).
        t0 = time.perf_counter()
        for _ in range(3):
            packed_k, vec_norm = encode_keys(K, cfg)
        t1 = time.perf_counter()
        enc_k_mts = 3 * n_tokens / (t1 - t0) / 1e6

        # Time V encode (min-max scan + uniform quant + pack).
        t0 = time.perf_counter()
        for _ in range(3):
            packed_v, v_scale, v_zero = encode_values(V, cfg)
        t1 = time.perf_counter()
        enc_v_mts = 3 * n_tokens / (t1 - t0) / 1e6

        # Time score (q rot + LUT + dot product).
        t0 = time.perf_counter()
        for _ in range(3):
            _ = score_in_rotated_space(Q, packed_k, vec_norm, cfg, H=H)
        t1 = time.perf_counter()
        score_mts = 3 * n_tokens / (t1 - t0) / 1e6

        # Time V decode (unpack + uniform dequant).
        t0 = time.perf_counter()
        for _ in range(3):
            _ = decode_values(packed_v, v_scale, v_zero, cfg)
        t1 = time.perf_counter()
        dec_v_mts = 3 * n_tokens / (t1 - t0) / 1e6

        out_lines.append(
            f"{preset:<22} {enc_k_mts:>12.2f} {enc_v_mts:>12.2f} {score_mts:>12.2f} {dec_v_mts:>12.2f}"
        )

    return "\n".join(out_lines)


# ----------------------------------------------------------------------------
# Driver.
# ----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[2048, 8192, 32768, 131072],
    )
    parser.add_argument("--n-keys", type=int, default=1024)
    parser.add_argument("--n-queries", type=int, default=128)
    parser.add_argument("--n-tokens-throughput", type=int, default=16384)
    args = parser.parse_args()

    print("=" * 78)
    print("TurboQuant benchmark vs fp16 KV cache")
    print("=" * 78)

    print(benchmark_memory(args.seq_lens, head_dim=args.head_dim))
    print(benchmark_memory_per_slot())
    print(benchmark_accuracy(args.head_dim, args.n_keys, args.n_queries))
    print(benchmark_throughput(args.head_dim, args.n_tokens_throughput))

    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()

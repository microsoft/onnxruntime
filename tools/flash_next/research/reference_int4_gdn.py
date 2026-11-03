import argparse
import json
from pathlib import Path

import numpy as np

from capture_int4_qmoe import sha256


def recurrent_reference(packed, decay, beta, a_log, dt_bias, initial):
    state = initial.astype(np.float64).copy()
    if state.ndim != 4 or state.shape[0] != 1:
        raise ValueError("This diagnostic supports batch1 V-major states")
    heads_v, size_v, size_k = state.shape[1:]
    flat = packed.reshape(-1, packed.shape[-1]).astype(np.float64)
    query_width = (flat.shape[1] - heads_v * size_v) // 2
    heads_q = query_width // size_k
    if not heads_q or heads_v % heads_q or 2 * query_width + heads_v * size_v != flat.shape[1]:
        raise ValueError("Invalid packed QKV shape")
    query = np.repeat(flat[:, :query_width].reshape(-1, heads_q, size_k), heads_v // heads_q, axis=1)
    key = np.repeat(flat[:, query_width:2 * query_width].reshape(-1, heads_q, size_k),
                    heads_v // heads_q, axis=1)
    query /= np.sqrt(np.sum(query * query, axis=-1, keepdims=True) + 1e-12)
    key /= np.sqrt(np.sum(key * key, axis=-1, keepdims=True) + 1e-12)
    value = flat[:, 2 * query_width:].reshape(-1, heads_v, size_v)
    gate = -np.exp(a_log.astype(np.float64)) * np.logaddexp(
        0, decay.reshape(-1, heads_v).astype(np.float64) + dt_bias.astype(np.float64))
    raw_beta = beta.reshape(-1, heads_v).astype(np.float64)
    mixing = np.exp(-np.logaddexp(0, -raw_beta))
    result = np.empty_like(value)
    for token in range(flat.shape[0]):
        state[0] *= np.exp(gate[token])[:, None, None]
        retrieved = np.einsum("hvk,hk->hv", state[0], key[token])
        delta = mixing[token, :, None] * (value[token] - retrieved)
        state[0] += delta[:, :, None] * key[token, :, None, :]
        result[token] = np.einsum("hvk,hk->hv", state[0], query[token]) / np.sqrt(size_k)
    return result.astype(np.float16), state.astype(np.float32)


def metrics(actual, expected):
    actual, expected = actual.astype(np.float64), expected.astype(np.float64)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Reference has a shape mismatch or nonfinite values")
    difference = actual - expected
    return {
        "elements": int(actual.size), "max_abs": float(np.abs(difference).max()),
        "rms_error": float(np.sqrt(np.mean(difference ** 2))),
        "reference_rms": float(np.sqrt(np.mean(expected ** 2))),
        "exact_elements": int((actual == expected).sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads(args.capture.with_suffix(".json").read_text())
    if metadata["capture_sha256"] != sha256(args.capture):
        raise ValueError("Capture identity changed")
    rows = []
    with np.load(args.capture) as data:
        initial = np.zeros_like(data["gdn_state_0"])
        for step in range(2):
            expected, state = recurrent_reference(
                *[data[f"gdn_{name}_{step}"] for name in ("packed", "decay", "beta", "a_log", "dt_bias")],
                initial)
            actual = data[f"gdn_output_{step}"].reshape(expected.shape)
            rows.append({"step": step, "output": metrics(actual, expected),
                         "state": metrics(data[f"gdn_state_{step}"], state)})
            initial = data[f"gdn_state_{step}"]
    report = {
        "status": "characterized_not_quality_qualification",
        "capture_sha256": sha256(args.capture), "rows": rows,
        "reference": "Independent float64 V-major gated-delta recurrence; FP16 output/F32 state; "
                     "Qwen gates, sigmoid beta, Q/K L2 normalization, contiguous 3:1 head expansion",
        "initial_state": "Zero for the first prefill; prior captured native state for the next decode",
        "caveat": "Does not reproduce exact GPU reduction/rounding or establish whole-model correctness",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

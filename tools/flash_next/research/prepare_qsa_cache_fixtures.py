from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def leading_rope(values, cosine, sine):
    result = values.copy()
    rotary_width = cosine.shape[-1]
    half = rotary_width // 2
    paired = np.concatenate([-values[..., half:rotary_width], values[..., :half]], axis=-1)
    result[..., :rotary_width] = values[..., :rotary_width] * cosine + paired * sine
    return result


root = Path(__file__).resolve().parent / "qsa-parallel-fixtures"
root.mkdir(exist_ok=True)
cases = [
    (0, 1, 1, False, False), (767, 1, 1, False, False), (768, 1, 1, False, False),
    (3071, 1, 4, False, False), (3072, 1, 4, False, False),
    (3079, 2, 32, False, False), (8192, 1, 4, False, False),
    (32768, 1, 4, False, False), (8192, 2, 4, True, True),
    (32768, 2, 4, True, False), (8192, 1, 4, False, True),
]
cases = [(*case, "holes") for case in cases] + [
    (8192, 1, 32, True, True, "holes"), (32768, 1, 32, True, False, "holes"),
    (8193, 2, 4, True, False, "empty"), (8192, 1, 4, False, False, "none"),
    (32768, 2, 4, True, True, "none"),
    (767, 1, 1, False, False, "none"), (768, 1, 1, False, False, "none"),
    (3072, 1, 4, False, False, "none"),
    (8192, 2, 32, True, True, "rotary"), (32768, 1, 4, False, False, "rotary"),
    (8192, 1, 32, True, False, "partial_rotary"),
]
cases = [(*case, case[2] == 32 and case[0] >= 8192) for case in cases] + [
    (31, 1, 32, True, True, "holes", True),
    (32, 1, 32, True, False, "none", True),
    (1024, 1, 32, True, True, "holes", True),
    (2048, 1, 32, True, False, "holes", True),
    (2048, 1, 32, False, True, "holes", True),
    (2048, 16, 32, True, True, "holes", True),
    (2048, 1, 32, True, False, "empty", True),
    (2048, 1, 32, True, False, "partial_rotary", True),
]
cases = [(*case[:6], 16 if case[6] else 2) for case in cases] + [
    (31, 1, 4, True, True, "model_rotary", 4),
    (32, 1, 4, True, True, "model_rotary", 4),
    (2048, 1, 4, True, True, "model_rotary", 4),
    (2048, 1, 4, False, False, "model_rotary", 4),
    (2048, 16, 4, True, True, "model_rotary", 4),
    (8192, 1, 4, True, True, "model_rotary", 4),
    (32768, 1, 4, True, True, "model_rotary", 4),
    (2048, 1, 4, True, True, "model_empty", 4),
    (2048, 1, 4, True, True, "model_none", 4),
    (31, 1, 4, True, True, "model_small_budget", 4),
    (8192, 1, 4, True, True, "model_irregular_budget", 4),
]
cases += [
    (past_length, 16, 4, half, half, mask_mode, 4)
    for past_length in (16364, 16368, 16372, 32768)
    for half, mask_mode in ((True, "model_rotary"), (False, "model_none"))
]
for past_length, rows, ratio, half, packed, mask_mode, heads in cases:
    realistic = heads >= 4
    batch, width = 1 if mask_mode.startswith("model_") else 2, 128 if realistic else 4
    total = past_length + rows
    dtype = np.float16 if half else np.float32
    tensor_type = TensorProto.FLOAT16 if half else TensorProto.FLOAT
    pattern = np.resize(np.array([1, .5, -.25, .125], dtype=dtype), width)
    query = np.tile(pattern, (batch, rows, heads)).reshape(batch, rows, heads * width)
    keys = np.zeros((batch, total, width), dtype=dtype)
    for token in range(total):
        keys[:, token, (token // ratio) % width] = 1
    mask = np.ones((batch, total), dtype=np.int64)
    if mask_mode in ("holes", "rotary", "partial_rotary", "model_rotary",
                     "model_small_budget", "model_irregular_budget"):
        for b in range(batch):
            for token in (b, 8, 53, 2050, 3011, 8190, total - 3):
                if 0 <= token < total:
                    mask[b, token] = 0
            if total > 11:
                mask[b, 11] = 1 << 40
    elif mask_mode in ("empty", "model_empty"):
        mask.fill(0)
    budget = (12 if mask_mode == "model_small_budget" else
              2052 if mask_mode == "model_irregular_budget" else 2048 if realistic else 2 * ratio)
    capacity = budget + ratio - 1
    selected = np.full((batch, rows, capacity), -1, dtype=np.int32)
    rotary = mask_mode in ("rotary", "partial_rotary", "model_rotary",
                          "model_small_budget", "model_irregular_budget")
    rotary_width = 32 if mask_mode.startswith("model_") else width // 2 if mask_mode == "partial_rotary" else width
    q_weight = np.resize(np.array([1, 2, 4, 8], dtype=dtype), width) if rotary else np.ones(width, dtype=dtype)
    k_weight = np.resize(np.array([.5, .25, .125, .0625], dtype=dtype), width) if rotary else np.ones(width, dtype=dtype)
    cos = np.ones((total, rotary_width), dtype=dtype)
    sin = np.zeros((total, rotary_width), dtype=dtype)
    if rotary:
        cos[:] = np.array([1, 0, -1, 0], dtype=dtype)[np.arange(total) % 4, None]
        sin[:] = np.array([0, 1, 0, -1], dtype=dtype)[np.arange(total) % 4, None]
    q = query.reshape(batch, rows, heads, width).astype(np.float64)
    q /= np.sqrt(np.mean(q * q, axis=-1, keepdims=True) + 1e-6)
    q *= q_weight
    for b in range(batch):
        for row in range(rows):
            visible = np.flatnonzero(mask[b, :past_length + row + 1])
            block_count = len(visible) // ratio
            pooled = np.array([np.mean(keys[b, visible[i * ratio:(i + 1) * ratio]].astype(np.float64), axis=0)
                               for i in range(block_count)]).reshape(block_count, width)
            pooled /= np.sqrt(np.mean(pooled * pooled, axis=-1, keepdims=True) + 1e-6)
            pooled *= k_weight
            positions = visible[:block_count * ratio:ratio]
            pooled = leading_rope(pooled, cos[positions].astype(np.float64), sin[positions].astype(np.float64))
            rotated_query = leading_rope(q[b, row], cos[past_length + row].astype(np.float64),
                                         sin[past_length + row].astype(np.float64))
            scores = np.maximum(rotated_query @ pooled.T, 0).sum(axis=0) * .125
            order = np.lexsort((np.arange(block_count), -scores))[:budget // ratio]
            indices = [int(token) for block in order for token in visible[block * ratio:(block + 1) * ratio]]
            indices.extend(int(token) for token in visible[block_count * ratio:])
            selected[b, row, :len(indices)] = indices
    values = {
        "query": np.concatenate([query, keys[:, past_length:]], axis=-1) if packed else query,
        "query_norm": q_weight, "key_norm": k_weight,
        "cos": cos, "sin": sin,
        "mask": mask, "past_key": keys[:, :past_length],
    }
    if not packed:
        values["key"] = keys[:, past_length:]
    if mask_mode in ("none", "model_none"):
        del values["mask"]
    inputs = ["query", "" if packed else "key", "query_norm", "key_norm", "cos", "sin",
              "" if mask_mode in ("none", "model_none") else "mask", "past_key"]
    outputs = [helper.make_tensor_value_info("indices", TensorProto.INT32, [batch, rows, capacity]),
               helper.make_tensor_value_info("present", tensor_type, [batch, total, width])]
    node = helper.make_node("SparseAttentionIndexer", inputs, ["indices", "present"], domain="com.microsoft",
                            policy_mode="qsa", compress_ratio=ratio, token_budget=budget, scale=.125)
    name = (f"qsa_past{past_length}_rows{rows}_ratio{ratio}_half{int(half)}_packed{int(packed)}"
            f"_h{heads}_d{width}_budget{budget}_{mask_mode}")
    model = helper.make_model(helper.make_graph(
        [node], name, [], outputs, [numpy_helper.from_array(value, key) for key, value in values.items()]),
        opset_imports=[helper.make_opsetid("", 22), helper.make_opsetid("com.microsoft", 1)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, root / f"{name}.onnx")
    reference = helper.make_model(helper.make_graph(
        [helper.make_node("Identity", ["expected_indices"], ["indices"]),
         helper.make_node("Identity", ["expected_present"], ["present"])],
        "reference", [], outputs, [numpy_helper.from_array(selected, "expected_indices"),
                                    numpy_helper.from_array(keys, "expected_present")]),
        opset_imports=[helper.make_opsetid("", 22)])
    reference.ir_version = 10
    onnx.save(reference, root / f"{name}_reference.onnx")
print(f"Prepared {len(cases)} independent masked, causal, stable-ranking QSA fixtures in {root}")

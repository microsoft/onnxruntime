import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from qualify_quickgelu_mul_long_context import sha


ROOT = Path(__file__).parent
DIRECTORY = ROOT / "qsa-prefill-score16-fixtures-v1"
CASES = ((0, 1024, True, False), (2048, 512, True, True), (8192, 512, True, True),
         (32768, 1024, True, True), (8192, 511, True, False), (8192, 1025, True, False),
         (8192, 512, False, False))


def rotate(values, cosine, sine):
    result = values.copy()
    half = cosine.shape[-1] // 2
    left, right = values[..., :half], values[..., half:half * 2]
    result[..., :half] = left * cosine[..., :half] - right * sine[..., :half]
    result[..., half:half * 2] = right * cosine[..., half:] + left * sine[..., half:]
    return result


def reference_indices(query, keys, mask, q_weight, k_weight, cosine, sine, past, rows):
    visible = np.flatnonzero(mask)
    blocks = len(visible) // 4
    pooled = keys[visible[:blocks * 4].reshape(blocks, 4)].astype(np.float64).mean(axis=1)
    pooled /= np.sqrt(np.mean(pooled * pooled, axis=-1, keepdims=True) + 1e-6)
    pooled *= k_weight
    pooled = rotate(pooled, cosine[visible[:blocks * 4:4]].astype(np.float64),
                    sine[visible[:blocks * 4:4]].astype(np.float64))
    q = query.astype(np.float64)
    q /= np.sqrt(np.mean(q * q, axis=-1, keepdims=True) + 1e-6)
    q *= q_weight
    phase_scores = []
    for phase in range(min(4, len(cosine))):
        rotated = rotate(q, cosine[phase].astype(np.float64), sine[phase].astype(np.float64))
        phase_scores.append(np.maximum(rotated @ pooled.T, 0).sum(axis=0) * .125)
    output = np.full((1, rows, 2051), -1, dtype=np.int32)
    for row in range(rows):
        count = int(np.searchsorted(visible, past + row + 1))
        block_count = count // 4
        scores = phase_scores[(past + row) % 4][:block_count]
        order = np.lexsort((np.arange(block_count), -scores))[:512]
        selected = visible[(order[:, None] * 4 + np.arange(4)).reshape(-1)]
        tail = visible[block_count * 4:count]
        indices = np.concatenate((selected, tail))
        output[0, row, :len(indices)] = indices
    return output


def main():
    DIRECTORY.mkdir(exist_ok=False)
    records = []
    for past, rows, half, holes in CASES:
        dtype, tensor_type = (np.float16, TensorProto.FLOAT16) if half else (np.float32, TensorProto.FLOAT)
        total = past + rows
        pattern = np.resize(np.array([1, .5, -.25, .125], dtype=dtype), 128)
        q = np.tile(pattern, (4, 1))
        keys = np.zeros((total, 128), dtype=dtype)
        tokens = np.arange(total)
        keys[tokens, (tokens // 4) % 128] = 1
        mask = np.ones(total, dtype=np.int64)
        if holes:
            for token in (0, 8, 53, 2050, 3011, 8190, total - 3):
                if token < total:
                    mask[token] = 0
            mask[11] = 1 << 40
        cosine, sine = np.ones((total, 32), dtype=dtype), np.zeros((total, 32), dtype=dtype)
        if holes:
            cosine[:] = np.array([1, 0, -1, 0], dtype=dtype)[tokens % 4, None]
            sine[:] = np.array([0, 1, 0, -1], dtype=dtype)[tokens % 4, None]
        q_weight = np.resize(np.array([1, 2, 4, 8], dtype=dtype), 128) if holes else np.ones(128, dtype=dtype)
        k_weight = np.resize(np.array([.5, .25, .125, .0625], dtype=dtype), 128) if holes else np.ones(128, dtype=dtype)
        expected = reference_indices(q, keys, mask, q_weight, k_weight, cosine, sine, past, rows)
        packed = np.concatenate((np.tile(q.reshape(1, 1, 512), (1, rows, 1)),
                                 keys[None, past:]), axis=-1)
        values = dict(query=packed, query_norm=q_weight, key_norm=k_weight, cos=cosine, sin=sine,
                      past_key=keys[None, :past])
        if holes:
            values["mask"] = mask[None]
        outputs = [helper.make_tensor_value_info("indices", TensorProto.INT32, [1, rows, 2051]),
                   helper.make_tensor_value_info("present", tensor_type, [1, total, 128])]
        name = f"qsa_prefill_past{past}_rows{rows}_half{int(half)}_holes{int(holes)}"
        model = helper.make_model(helper.make_graph(
            [helper.make_node("SparseAttentionIndexer",
                ["query", "", "query_norm", "key_norm", "cos", "sin", "mask" if holes else "", "past_key"],
                ["indices", "present"], domain="com.microsoft", policy_mode="qsa",
                compress_ratio=4, token_budget=2048, scale=.125)],
            name, [], outputs, [numpy_helper.from_array(v, k) for k, v in values.items()]),
            opset_imports=[helper.make_opsetid("", 22), helper.make_opsetid("com.microsoft", 1)])
        model.ir_version = 10
        onnx.checker.check_model(model)
        path = DIRECTORY / f"{name}.onnx"
        onnx.save(model, path)
        reference = helper.make_model(helper.make_graph(
            [helper.make_node("Identity", ["expected_indices"], ["indices"]),
             helper.make_node("Identity", ["expected_present"], ["present"])],
            name + "_reference", [], outputs,
            [numpy_helper.from_array(expected, "expected_indices"),
             numpy_helper.from_array(keys[None], "expected_present")]),
            opset_imports=[helper.make_opsetid("", 22)])
        reference.ir_version = 10
        onnx.checker.check_model(reference)
        reference_path = DIRECTORY / f"{name}_reference.onnx"
        onnx.save(reference, reference_path)
        records.append(dict(path=str(path), sha256=sha(path), reference_path=str(reference_path),
                            reference_sha256=sha(reference_path), past=past, rows=rows,
                            half=half, holes=holes, candidate_admitted=half and rows in (512, 1024)))
    with (DIRECTORY / "manifest.json").open("x") as stream:
        stream.write(json.dumps(dict(fixtures=records, source_sha256=sha(Path(__file__))), indent=2) + "\n")
    print(json.dumps(dict(fixtures=len(records), admitted=sum(r["candidate_admitted"] for r in records))))


if __name__ == "__main__":
    main()

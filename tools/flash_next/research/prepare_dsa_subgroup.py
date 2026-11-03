import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def prepare_case(out, name, *, dtype=np.float16, head_size=128, heads=4, kv_heads=2,
                 batch=1, sequence=1, past=15, selected=17, auxiliary=0,
                 shared_auxiliary=False, local_window=0, sink=False, smooth=False,
                 empty=False, check_cache=True):
    rng = np.random.default_rng(541)
    capacity = past + sequence + 3
    random = lambda shape: rng.normal(0, 0.3, shape).astype(dtype)
    q = random((batch, sequence, heads, head_size))
    k = random((batch, sequence, kv_heads, head_size))
    v = random(k.shape)
    pk = random((batch, kv_heads, capacity, head_size))
    pv = random(pk.shape)
    ak = random((batch, kv_heads, auxiliary, head_size))
    av = ak if shared_auxiliary else random(ak.shape)
    indices = rng.integers(-2, (auxiliary or past + sequence) + 3,
                           (batch, sequence, selected), dtype=np.int32)
    if selected:
        indices[:, :, :min(selected, 3)] = np.arange(min(selected, 3), dtype=np.int32)
    counts = np.full((batch, sequence), 0 if empty else selected, np.int32)
    heads_sink = np.linspace(-1, 1, heads, dtype=dtype)
    appended_k, appended_v = pk.copy(), pv.copy()
    appended_k[:, :, past:past + sequence] = k.transpose(0, 2, 1, 3)
    appended_v[:, :, past:past + sequence] = v.transpose(0, 2, 1, 3)
    scale = float(np.float32(head_size ** -0.5))
    expected = np.zeros(q.shape, dtype=dtype)
    for b in range(batch):
        for s in range(sequence):
            position = past + s
            for h in range(heads):
                kh = h // (heads // kv_heads)
                keys, values = [], []
                for index in range(max(0, position - local_window + 1), position + 1) if local_window else []:
                    keys.append(appended_k[b, kh, index])
                    values.append(appended_v[b, kh, index])
                for index in indices[b, s, :counts[b, s]]:
                    if auxiliary and 0 <= index < auxiliary:
                        keys.append(ak[b, kh, index])
                        values.append(av[b, kh, index])
                    elif not auxiliary and 0 <= index <= position:
                        keys.append(appended_k[b, kh, index])
                        values.append(appended_v[b, kh, index])
                if not keys:
                    continue
                logits = np.asarray(keys, np.float64) @ q[b, s, h].astype(np.float64) * scale
                if sink or smooth:
                    logits = np.append(logits, float(heads_sink[h]) if sink else 0.0)
                    values.append(np.zeros(head_size, dtype=dtype))
                weights = np.exp(logits - np.max(logits))
                expected[b, s, h] = (weights @ np.asarray(values, np.float64) / weights.sum()).astype(dtype)
    tensors = []

    def tensor(name, value):
        tensors.append(numpy_helper.from_array(value, name))
        return name

    inputs = [
        tensor("q", q.reshape(batch, sequence, -1)),
        tensor("k", k.reshape(batch, sequence, -1)),
        tensor("v", v.reshape(batch, sequence, -1)),
        tensor("pk", pk), tensor("pv", pv),
        tensor("ak", ak) if auxiliary else "",
        tensor("av", av) if auxiliary and not shared_auxiliary else "",
        tensor("indices", indices.reshape(batch * sequence, selected)),
        tensor("counts", counts.reshape(-1)),
        tensor("seqlens", np.full(batch, past + sequence - 1, np.int32)),
        tensor("total", np.array(past + sequence, np.int32)),
        "", "", "", "", "", tensor("sink", heads_sink) if sink else "",
    ]
    node = helper.make_node(
        "DynamicSparseAttention", inputs, ["attention", "present_k", "present_v"],
        domain="com.microsoft", num_heads=heads, kv_num_heads=kv_heads, scale=scale,
        is_causal=1, attention_mode="local_plus_selected" if local_window else "selected_only",
        selected_kv_source="auxiliary" if auxiliary else "main",
        local_window_size=local_window or -1, auxiliary_kv_shared=int(shared_auxiliary),
        smooth_softmax=int(smooth))
    nodes = [node]
    if check_cache:
        tensor("flat", np.array([-1], np.int64))
        for src in ["attention", "present_k", "present_v"]:
            nodes.append(helper.make_node("Reshape", [src, "flat"], [src + "_flat"]))
        nodes.append(helper.make_node("Concat", ["attention_flat", "present_k_flat", "present_v_flat"],
                                      ["y"], axis=0))
        expected = np.concatenate([expected.reshape(-1), appended_k.reshape(-1), appended_v.reshape(-1)])
    else:
        nodes.append(helper.make_node("Identity", ["attention"], ["y"]))
        expected = expected.reshape(batch, sequence, heads * head_size)
    element_type = TensorProto.FLOAT16 if dtype == np.float16 else TensorProto.FLOAT
    output = helper.make_tensor_value_info("y", element_type, list(expected.shape))
    model = helper.make_model(helper.make_graph(nodes, name, [], [output], tensors),
                              opset_imports=[helper.make_opsetid("", 22), helper.make_opsetid("com.microsoft", 1)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, out / f"{name}.onnx")
    reference = helper.make_model(helper.make_graph(
        [helper.make_node("Identity", ["expected"], ["y"])], name + "_reference", [], [output],
        [numpy_helper.from_array(expected, "expected")]), opset_imports=[helper.make_opsetid("", 22)])
    reference.ir_version = 10
    onnx.save(reference, out / f"{name}_reference.onnx")
    print(name, "elements", expected.size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "dsa-subgroup-fixtures")
    out = parser.parse_args().output
    out.mkdir(exist_ok=True)
    for dtype in (np.float16, np.float32):
        tag = np.dtype(dtype).name
        for size in (8, 32, 64, 128, 256):
            prepare_case(out, f"selected_{tag}_d{size}", dtype=dtype, head_size=size)
        prepare_case(out, f"empty_{tag}", dtype=dtype, empty=True)
        prepare_case(out, f"zero_width_{tag}", dtype=dtype, selected=0, empty=True)
        prepare_case(out, f"aux_shared_sink_{tag}", dtype=dtype, auxiliary=19, shared_auxiliary=True,
                     local_window=4, sink=True)
        prepare_case(out, f"aux_distinct_smooth_{tag}", dtype=dtype, auxiliary=23,
                     local_window=8, smooth=True, batch=2)
        prepare_case(out, f"prefill_fallback_{tag}", dtype=dtype, sequence=2, smooth=True)
    prepare_case(out, "model_2k", heads=24, kv_heads=2, head_size=256, past=2048,
                 selected=2051, check_cache=False)
    for dtype in (np.float16, np.float32):
        tag = np.dtype(dtype).name
        prepare_case(out, f"split_aux_shared_sink_{tag}", dtype=dtype, selected=257,
                     auxiliary=129, shared_auxiliary=True, local_window=128, sink=True)
        prepare_case(out, f"split_aux_distinct_smooth_{tag}", dtype=dtype, selected=257,
                     auxiliary=129, local_window=128, smooth=True)
        prepare_case(out, f"split_empty_{tag}", dtype=dtype, selected=257, empty=True)

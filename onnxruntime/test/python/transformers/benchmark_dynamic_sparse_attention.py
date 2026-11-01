# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Op-level benchmark of the INT8 KV-cache DynamicSparseAttention CUDA kernels.

The shapes are those of the 12 full-attention layers of Qwen3.8-Flash-Next: 24 query heads, 2 KV heads, head size 256,
INT8 KV cache with per-channel scales, softmax scale 1/16 and 2048 selected candidates per query row. Each
(rows, context length) case becomes a single-node model, and each kernel configuration becomes its own session,
because the ep.cuda.dsa_* options are read when the kernel is created:

  rows >= 9 (prefill)  legacy: ep.cuda.dsa_prefill_int8_tc=0, the BF16-dequantizing kernel
                       tc:     ep.cuda.dsa_prefill_int8_tc=1, the tensor-core kernel
  rows <= 8 (decode)   legacy: ep.cuda.dsa_decode_int8_tc=0, the CUDA-core grouped kernel
                       tc:     ep.cuda.dsa_decode_int8_tc=1, the split-K tensor-core kernel, once for each
                               --tiles-per-split value (0 = automatic)

The time of a configuration is the GPU time of the CUDA kernels one Run launches, measured with the CUPTI-based torch
profiler, so host launch overhead does not enter. "attention" sums the DynamicSparseAttention kernels (including the
decode merge kernel), "total" all GPU activity of the node (cache append, query preparation, ...). A round measures
every configuration once, back to back, and the order alternates between rounds. The speedup of a configuration is the
median over rounds of the paired ratio to the first (legacy) configuration, which cancels clock and temperature drift.
The "rms err" column is the RMS error of the BF16 output relative to the RMS of an FP64 reference, computed over
--reference-rows evenly spaced query rows (0 disables it). It is the error against the exact result, not the
difference to the legacy kernel, which has errors of its own.

Example (a CUDA plugin EP build needs ORT_TEST_CUDA_PLUGIN_EP=1 and, unless the wheel carries the freshly built
library, ORT_CUDA_PLUGIN_PATH):

  python benchmark_dynamic_sparse_attention.py --rows 8 --context 2048 8192 32768 --tiles-per-split 0 1 2 4 8 16
  python benchmark_dynamic_sparse_attention.py --rows 128 2048 8192 --context 8192 32768 --json prefill.json

The key and value scales are synthetic (uniform in the ranges of the Qwen3.8-Flash-Next layers) unless --rc-model names
a model whose DynamicSparseAttention node supplies them. The multipliers below reproduce the slow FP32 fallback of the
tensor-core kernels (queries or value scales far outside the usual range).
"""

import argparse
import json
import platform
import statistics
import sys

import numpy as np
import onnx
import torch
from cuda_plugin_ep_helper import CUDA_PLUGIN_EP_NAME, ensure_cuda_plugin_ep_registered
from onnx import TensorProto, helper, numpy_helper

import onnxruntime as ort

NUM_HEADS = 24
KV_HEADS = 2
HEAD_SIZE = 256
SOFTMAX_SCALE = 0.0625
MAX_DECODE_ROWS = 8
TORCH_TO_ONNX = {
    torch.bfloat16: TensorProto.BFLOAT16,
    torch.int8: TensorProto.INT8,
    torch.int32: TensorProto.INT32,
}


def load_rc_scales(model_path, layer):
    """Returns the INT8 key and value scales of the layer-th DynamicSparseAttention node of a model."""
    model = onnx.load(model_path, load_external_data=False)
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    nodes = [node for node in model.graph.node if node.op_type == "DynamicSparseAttention"]
    if layer >= len(nodes):
        raise ValueError(f"{model_path} has {len(nodes)} DynamicSparseAttention nodes, --layer {layer} is out of range")
    return tuple(
        numpy_helper.to_array(initializers[nodes[layer].input[index]]).astype(np.float32) for index in (18, 19)
    )


def make_scales(args):
    if args.rc_model:
        key_scale, value_scale = load_rc_scales(args.rc_model, args.layer)
    else:
        rng = np.random.default_rng(args.seed)
        key_scale = rng.uniform(0.002, 0.07, KV_HEADS * HEAD_SIZE).astype(np.float32)
        value_scale = rng.uniform(0.002, 0.03, KV_HEADS * HEAD_SIZE).astype(np.float32)
    return key_scale * np.float32(args.key_scale_multiplier), value_scale * np.float32(args.value_scale_multiplier)


def make_model(rows, context, max_selected, key_scale, value_scale):
    def value_info(name, element_type, shape):
        return helper.make_tensor_value_info(name, element_type, shape)

    cache_shape = [1, KV_HEADS, context, HEAD_SIZE]
    graph = helper.make_graph(
        [
            helper.make_node(
                "DynamicSparseAttention",
                [
                    "query",
                    "key",
                    "value",
                    "past_key",
                    "past_value",
                    "",
                    "",
                    "selected_indices",
                    "selected_counts",
                    "seqlens_k",
                    "total_sequence_length",
                    *[""] * 7,
                    "key_scale",
                    "value_scale",
                ],
                ["output", "present_key", "present_value"],
                domain="com.microsoft",
                num_heads=NUM_HEADS,
                kv_num_heads=KV_HEADS,
                scale=SOFTMAX_SCALE,
            )
        ],
        "dynamic_sparse_attention",
        [
            value_info("query", TensorProto.BFLOAT16, [1, rows, NUM_HEADS * HEAD_SIZE]),
            value_info("key", TensorProto.BFLOAT16, [1, rows, KV_HEADS * HEAD_SIZE]),
            value_info("value", TensorProto.BFLOAT16, [1, rows, KV_HEADS * HEAD_SIZE]),
            value_info("past_key", TensorProto.INT8, cache_shape),
            value_info("past_value", TensorProto.INT8, cache_shape),
            value_info("selected_indices", TensorProto.INT32, [rows, max_selected]),
            value_info("selected_counts", TensorProto.INT32, [rows]),
            value_info("seqlens_k", TensorProto.INT32, [1]),
        ],
        [
            value_info("output", TensorProto.BFLOAT16, [1, rows, NUM_HEADS * HEAD_SIZE]),
            value_info("present_key", TensorProto.INT8, cache_shape),
            value_info("present_value", TensorProto.INT8, cache_shape),
        ],
        [
            numpy_helper.from_array(np.array(context, dtype=np.int32), "total_sequence_length"),
            numpy_helper.from_array(key_scale, "key_scale"),
            numpy_helper.from_array(value_scale, "value_scale"),
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)]
    )
    model.ir_version = 10
    return model.SerializeToString()


def make_tensors(rows, context, max_selected, args):
    """The query rows are the last `rows` positions of a context of `context` tokens; each selects random positions."""
    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed)

    def normal(*shape, std):
        return torch.randn(*shape, device="cuda", generator=generator) * std

    def cache_codes():
        return normal(1, KV_HEADS, context, HEAD_SIZE, std=args.cache_code_std).clamp(-127, 127).round().to(torch.int8)

    positions = torch.arange(context - rows, context, device="cuda")
    counts = (positions + 1).clamp(max=max_selected).to(torch.int32)
    selected = torch.empty(rows, max_selected, dtype=torch.int32, device="cuda")
    columns = torch.arange(context, device="cuda")
    for start in range(0, rows, 1024):
        stop = min(start + 1024, rows)
        noise = torch.rand(stop - start, context, device="cuda", generator=generator)
        noise[columns[None, :] > positions[start:stop, None]] = -1.0
        selected[start:stop] = noise.topk(max_selected, dim=1).indices.to(torch.int32)
    selected[torch.arange(max_selected, device="cuda")[None, :] >= counts[:, None]] = -1
    return {
        "query": normal(1, rows, NUM_HEADS * HEAD_SIZE, std=args.query_std).to(torch.bfloat16),
        "key": normal(1, rows, KV_HEADS * HEAD_SIZE, std=1.0).to(torch.bfloat16),
        "value": normal(1, rows, KV_HEADS * HEAD_SIZE, std=1.0).to(torch.bfloat16),
        "past_key": cache_codes(),
        "past_value": cache_codes(),
        "selected_indices": selected,
        "selected_counts": counts,
        "seqlens_k": torch.tensor([context - 1], dtype=torch.int32, device="cuda"),
    }


def select_plugin_device(device_id):
    """The device of the CUDA plugin EP when ORT_TEST_CUDA_PLUGIN_EP=1 selects it, otherwise None."""
    if not ensure_cuda_plugin_ep_registered():
        return None
    for device in ort.get_ep_devices():
        if device.ep_name == CUDA_PLUGIN_EP_NAME:
            found = device.ep_options.get("device_id", device.ep_metadata.get("cuda_device_id"))
            if found is None or int(found) == device_id:
                return device
    return None


def create_session(model_bytes, options, args):
    session_options = ort.SessionOptions()
    for key, value in options.items():
        session_options.add_session_config_entry(key, value)
    plugin_device = select_plugin_device(args.device)
    if plugin_device is not None:
        session_options.add_provider_for_devices([plugin_device], {})
        return ort.InferenceSession(model_bytes, session_options)
    providers = [("CUDAExecutionProvider", {"device_id": args.device})]
    return ort.InferenceSession(model_bytes, session_options, providers=providers)


def create_binding(session, tensors, output, args):
    binding = session.io_binding()
    for name, tensor in tensors.items():
        binding.bind_input(
            name, "cuda", args.device, TORCH_TO_ONNX[tensor.dtype], list(tensor.shape), tensor.data_ptr()
        )
    # The cache is updated in place, like the model does with its past_key_values buffers.
    binding.bind_output("output", "cuda", args.device, TensorProto.BFLOAT16, list(output.shape), output.data_ptr())
    for name, cache in (("present_key", tensors["past_key"]), ("present_value", tensors["past_value"])):
        binding.bind_output(name, "cuda", args.device, TensorProto.INT8, list(cache.shape), cache.data_ptr())
    return binding


def reference_attention(tensors, key_scale, value_scale, rows_to_check):
    """FP64 attention of the given query rows over the cache the operator updated in place (new rows appended)."""
    scales = [
        torch.tensor(scale, device="cuda", dtype=torch.float64).view(KV_HEADS, 1, HEAD_SIZE)
        for scale in (key_scale, value_scale)
    ]
    keys = tensors["past_key"][0].double() * scales[0]
    values = tensors["past_value"][0].double() * scales[1]
    query = tensors["query"][0].double().view(-1, NUM_HEADS, HEAD_SIZE)
    group = NUM_HEADS // KV_HEADS
    reference = torch.empty(len(rows_to_check), NUM_HEADS, HEAD_SIZE, dtype=torch.float64, device="cuda")
    for position, row in enumerate(rows_to_check):
        selected = tensors["selected_indices"][row, : int(tensors["selected_counts"][row])].long()
        for kv_head in range(KV_HEADS):
            heads = slice(kv_head * group, (kv_head + 1) * group)
            scores = query[row, heads] @ keys[kv_head, selected].T * SOFTMAX_SCALE
            reference[position, heads] = torch.softmax(scores, dim=-1) @ values[kv_head, selected]
    return reference


def relative_rms_error(result, reference):
    return float((result - reference).pow(2).mean().sqrt() / reference.pow(2).mean().sqrt())


def short_kernel_name(name):
    head = name.replace("(anonymous namespace)::", "").split("(")[0].removeprefix("void ").strip()
    base, separator, template_arguments = head.partition("<")
    return base.split("::")[-1] + separator + template_arguments


def run_and_profile(session, binding, iterations):
    """Returns the GPU time per Run in microseconds: {kernel name: total}, summed over the profiled runs."""
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profile:
        for _ in range(iterations):
            session.run_with_iobinding(binding)
        binding.synchronize_outputs()
        torch.cuda.synchronize()
    times = {}
    for event in profile.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            name = short_kernel_name(event.name)
            times[name] = times.get(name, 0.0) + event.device_time / iterations
    return times


def configurations(rows, args):
    if rows <= MAX_DECODE_ROWS:
        configs = [("legacy (CUDA-core)", {"ep.cuda.dsa_decode_int8_tc": "0"})]
        for tiles in args.tiles_per_split:
            options = {
                "ep.cuda.dsa_decode_int8_tc": "1",
                "ep.cuda.dsa_decode_int8_tc_tiles_per_split": str(tiles),
            }
            configs.append((f"tc tiles_per_split={'auto' if tiles == 0 else tiles}", options))
        return configs
    return [
        ("legacy (BF16 dequantizing)", {"ep.cuda.dsa_prefill_int8_tc": "0"}),
        ("tc", {"ep.cuda.dsa_prefill_int8_tc": "1"}),
    ]


def format_range(values, digits):
    return f"{statistics.median(values):.{digits}f} [{min(values):.{digits}f}, {max(values):.{digits}f}]"


def run_case(rows, context, args, key_scale, value_scale):
    max_selected = min(args.selected, context)
    model_bytes = make_model(rows, context, max_selected, key_scale, value_scale)
    tensors = make_tensors(rows, context, max_selected, args)
    output = torch.empty(1, rows, NUM_HEADS * HEAD_SIZE, dtype=torch.bfloat16, device="cuda")
    configs = configurations(rows, args)
    iterations = args.iterations or (args.decode_iterations if rows <= MAX_DECODE_ROWS else args.prefill_iterations)

    rows_to_check = sorted({int(row) for row in np.linspace(0, rows - 1, min(rows, args.reference_rows)).round()})
    sessions, bindings, reference, rms_error = [], [], None, []
    for _, options in configs:
        session = create_session(model_bytes, options, args)
        binding = create_binding(session, tensors, output, args)
        for _ in range(args.warmup):
            session.run_with_iobinding(binding)
        binding.synchronize_outputs()
        torch.cuda.synchronize()
        if args.reference_rows > 0:
            if reference is None:
                # The first run has appended the new key/value rows to the cache; later runs rewrite the same values.
                reference = reference_attention(tensors, key_scale, value_scale, rows_to_check).view(
                    len(rows_to_check), -1
                )
            rms_error.append(relative_rms_error(output[0, rows_to_check].float(), reference.float()))
        else:
            rms_error.append(float("nan"))
        sessions.append(session)
        bindings.append(binding)

    attention = [[] for _ in configs]
    total = [[] for _ in configs]
    kernels = [{} for _ in configs]
    for round_index in range(args.rounds):
        order = range(len(configs)) if round_index % 2 == 0 else reversed(range(len(configs)))
        for index in order:
            for _ in range(args.warmup):
                sessions[index].run_with_iobinding(bindings[index])
            times = run_and_profile(sessions[index], bindings[index], iterations)
            attention[index].append(sum(t for name, t in times.items() if "DynamicSparseAttention" in name))
            total[index].append(sum(times.values()))
            kernels[index] = times

    kind = "decode" if rows <= MAX_DECODE_ROWS else "prefill"
    unit, scale, digits = ("us", 1.0, 2) if kind == "decode" else ("ms", 1e-3, 3)
    print(
        f"\n{kind}: rows={rows}, context={context}, selected={max_selected}, {iterations} runs x {args.rounds} rounds"
    )
    print(f"| configuration | attention {unit} | total {unit} | speedup | rms err |")
    print("|---|---|---|---|---|")
    results = []
    for index, (label, options) in enumerate(configs):
        ratios = [base / value for base, value in zip(attention[0], attention[index], strict=True)]
        attention_scaled = [value * scale for value in attention[index]]
        total_scaled = [value * scale for value in total[index]]
        print(
            f"| {label} | {format_range(attention_scaled, digits)} | {format_range(total_scaled, digits)} "
            f"| {format_range(ratios, 2)}x | {rms_error[index]:.2e} |"
        )
        results.append(
            {
                "label": label,
                "options": options,
                "attention_us": attention[index],
                "total_us": total[index],
                "kernels_us": kernels[index],
                "rms_error": rms_error[index],
            }
        )
        if args.breakdown:
            for name, value in sorted(kernels[index].items(), key=lambda item: -item[1])[:5]:
                print(f"|   {name[:90]} | {value:.2f} us | | | |")
    return {
        "kind": kind,
        "rows": rows,
        "context": context,
        "selected": max_selected,
        "iterations": iterations,
        "configurations": results,
    }


def environment():
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "onnxruntime": ort.__version__,
        "onnxruntime_build_info": ort.get_build_info(),
        "torch": torch.__version__,
        "gpu": properties.name,
        "sm_count": properties.multi_processor_count,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "python": platform.python_version(),
        "command": " ".join(sys.argv),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, nargs="+", default=[8, 128, 8192], help="query rows; <= 8 selects decode")
    parser.add_argument("--context", type=int, nargs="+", default=[8192], help="total sequence lengths")
    parser.add_argument("--selected", type=int, default=2048, help="selected candidates per row")
    parser.add_argument(
        "--tiles-per-split", type=int, nargs="+", default=[0], help="decode tc tiles per split (0=auto)"
    )
    parser.add_argument("--rounds", type=int, default=7, help="alternating rounds over all configurations")
    parser.add_argument("--iterations", type=int, default=None, help="profiled runs per round (default by kind)")
    parser.add_argument("--prefill-iterations", type=int, default=5)
    parser.add_argument("--decode-iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=3, help="unprofiled runs before each measurement")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference-rows", type=int, default=64, help="query rows checked against FP64 (0=skip)")
    parser.add_argument("--rc-model", default="", help="model whose DynamicSparseAttention node supplies the scales")
    parser.add_argument("--layer", type=int, default=0, help="index of that DynamicSparseAttention node")
    parser.add_argument("--query-std", type=float, default=1.4, help="BF16 query values ~ N(0, std^2); 1.4 is typical")
    parser.add_argument("--cache-code-std", type=float, default=40.0, help="INT8 cache codes ~ N(0, std^2), clipped")
    parser.add_argument("--key-scale-multiplier", type=float, default=1.0)
    parser.add_argument("--value-scale-multiplier", type=float, default=1.0, help="1.2e7 gives scales above 2^16")
    parser.add_argument("--breakdown", action="store_true", help="also print the largest kernels of each configuration")
    parser.add_argument("--json", default="", help="write the raw per-round timings and the environment here")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.cuda.set_device(args.device)
    key_scale, value_scale = make_scales(args)
    info = environment()
    print(json.dumps(info, indent=2))
    cases = []
    for rows in args.rows:
        for context in args.context:
            if rows > context:
                continue
            cases.append(run_case(rows, context, args, key_scale, value_scale))
    if args.json:
        with open(args.json, "w") as file:
            json.dump({"environment": info, "arguments": vars(args), "cases": cases}, file, indent=1)


if __name__ == "__main__":
    main()

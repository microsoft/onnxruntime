import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

import onnxruntime as ort


def digest(values):
    return hashlib.sha256(values.tobytes()).hexdigest()


def make_case(width, capacity, lengths, kind, dtype, packed=False, active=1):
    rng = np.random.default_rng(20261005)
    batch = len(lengths)
    tokens = width * batch
    past = np.array(lengths, dtype=np.int32)
    state_lengths = np.stack([past // 4, past % 4], axis=1)
    query = rng.standard_normal((tokens, 512)).astype(dtype)
    key = rng.standard_normal((tokens, 128)).astype(dtype)
    state = rng.standard_normal((batch, capacity, 128)).astype(dtype)
    if kind == "ties":
        query.fill(0)
    elif kind == "positive_ties":
        query.fill(1)
        state.fill(1)
    feeds = {
        "query": np.concatenate([query, key], axis=1) if packed else query,
        "qn": np.ones(128, dtype=dtype),
        "kn": np.ones(128, dtype=dtype),
        "cos": np.ones((max(lengths) + width + 4, 64), dtype=dtype),
        "sin": np.zeros((max(lengths) + width + 4, 64), dtype=dtype),
        "cu": np.arange(batch + 1, dtype=np.int32) * width,
        "past": past,
        "state": state,
        "buffer": rng.standard_normal((batch, 7, 128)).astype(dtype),
        "lengths": state_lengths,
        "capture": np.full(batch, width, dtype=np.int32),
        "active": np.array([active], dtype=np.int32),
    }
    if not packed:
        feeds["key"] = key
    inputs = ["query", "" if packed else "key", "qn", "kn", "cos", "sin", "cu", "past"]
    inputs += ["", "", "", "", "state", "buffer", "", "lengths", "capture", "active"]
    outputs = ["indices", "counts", "present_state", "present_buffer", "", "present_lengths", "update"]
    elem = TensorProto.FLOAT16 if dtype == np.float16 else TensorProto.FLOAT
    graph_inputs = [
        helper.make_tensor_value_info(name, TensorProto.INT32 if value.dtype == np.int32 else elem, list(value.shape))
        for name, value in feeds.items()
    ]
    output_shapes = [(tokens, 2051), (tokens,), (batch, capacity, 128), (batch, 7, 128), (batch, 2), (batch, 4, 128)]
    output_names = [name for name in outputs if name]
    graph_outputs = [
        helper.make_tensor_value_info(name, TensorProto.INT32 if index in (0, 1, 4) else elem, list(shape))
        for index, (name, shape) in enumerate(zip(output_names, output_shapes, strict=True))
    ]
    node = helper.make_node(
        "PackedSparseAttentionIndexer",
        inputs,
        outputs,
        domain="com.microsoft",
        policy_mode="qsa",
        compress_ratio=4,
        state_capacity=capacity,
        token_budget=2048,
        state_update_capacity=4,
    )
    model = helper.make_model(
        helper.make_graph([node], "packed_qsa_merge_parity", graph_inputs, graph_outputs),
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)],
        ir_version=10,
    )
    onnx.checker.check_model(model)
    return model, feeds


def run_case(device, model, feeds, graph):
    options = ort.SessionOptions()
    options.add_provider_for_devices([device], {"enable_cuda_graph": "1" if graph else "0"})
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(model.SerializeToString(), sess_options=options)
    assert "CUDAExecutionProvider" in session.get_providers(), session.get_providers()
    binding = session.io_binding()
    memory_info = device.memory_info(ort.OrtDeviceMemoryType.DEFAULT)
    values = {
        name: (
            ort.OrtValue.ortvalue_from_numpy(value)
            if name == "active"
            else ort.OrtValue.ortvalue_from_shape_and_type(value.shape, value.dtype, memory_info=memory_info)
        )
        for name, value in feeds.items()
    }
    for name, value in values.items():
        value.update_inplace(feeds[name])
        binding.bind_ortvalue_input(name, value)
    for output in session.get_outputs():
        binding.bind_output(output.name, "cuda", 0)
    session.run_with_iobinding(binding)
    results = binding.copy_outputs_to_cpu()
    if graph:
        session.run_with_iobinding(binding)
        replay = binding.copy_outputs_to_cpu()
        for expected, actual in zip(results, replay, strict=True):
            np.testing.assert_array_equal(actual, expected)
        past = feeds["past"].copy()
        past[0] = 0
        lengths = feeds["lengths"].copy()
        lengths[0] = 0
        values["past"].update_inplace(past)
        values["lengths"].update_inplace(lengths)
        session.run_with_iobinding(binding)
        changed = binding.copy_outputs_to_cpu()
        changed_feeds = dict(feeds, past=past, lengths=lengths)
        ordinary = run_case(device, model, changed_feeds, False)
        assert [digest(value) for value in changed] == ordinary
    if np.all(feeds["query"] == 0) or np.all(feeds["query"] == 1):
        for token, indices in enumerate(results[0]):
            request = token // (len(results[0]) // len(feeds["past"]))
            if results[1][token] >= 2048:
                np.testing.assert_array_equal(indices[:2048], np.arange(2048, dtype=np.int32))
    for request, past in enumerate(feeds["past"]):
        width = len(results[0]) // len(feeds["past"])
        if (int(past) + width) // 4 > feeds["state"].shape[1]:
            assert np.all(results[0][request * width : (request + 1) * width] == -1)
            assert np.all(results[1][request * width : (request + 1) * width] == 0)
            np.testing.assert_array_equal(results[2][request], feeds["state"][request])
            np.testing.assert_array_equal(results[3][request], feeds["buffer"][request])
            np.testing.assert_array_equal(results[4][request], feeds["lengths"][request])
    if feeds["active"][0] == 0:
        assert np.all(results[5] == 0)
    return [digest(value) for value in results]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument(
        "--allow-host-build-difference",
        action="store_true",
        help="Allow a non-companion host; reference comparisons still require identical host provenance",
    )
    parser.add_argument(
        "--graph-mode",
        choices=("strict", "separate", "skip"),
        default="strict",
        help="Separate records graph failures without blocking eager parity; skip omits graph checks",
    )
    args = parser.parse_args()
    ort.unregister_execution_provider_library("CUDAExecutionProvider")
    ort.register_execution_provider_library("qsa_parity", str(args.plugin.resolve()))
    devices = [device for device in ort.get_ep_devices() if device.ep_name == "CUDAExecutionProvider"]
    assert len(devices) == 1, "Expected only the explicitly registered CUDA plugin"
    mapped = {
        Path(line.split()[-1]) for line in Path("/proc/self/maps").read_text().splitlines() if "onnxruntime" in line
    }
    plugins = {path for path in mapped if path.name == "libonnxruntime_providers_cuda.so"}
    assert args.plugin.resolve() in plugins, plugins
    assert Path(devices[0].ep_metadata["library_path"]).resolve() == args.plugin.resolve()
    bindings = {path for path in mapped if path.name.startswith("onnxruntime_pybind11_state")}
    assert len(bindings) == 1, bindings
    binding = bindings.pop()
    binding_hash = hashlib.sha256(binding.read_bytes()).hexdigest()
    companion_binding = args.plugin.parent / "onnxruntime_pybind11_state.so"
    companion_binding_hash = hashlib.sha256(companion_binding.read_bytes()).hexdigest()
    cores = {path for path in mapped if path.name.startswith("libonnxruntime.so")}
    assert len(cores) == 1 or (args.allow_host_build_difference and not cores), cores
    core_embedded_in_binding = not cores
    core = cores.pop() if cores else binding
    core_hash = hashlib.sha256(core.read_bytes()).hexdigest()
    companion_core = args.plugin.parent / "libonnxruntime.so.1.31.0"
    companion_core_hash = hashlib.sha256(companion_core.read_bytes()).hexdigest()
    if not args.allow_host_build_difference:
        assert binding_hash == companion_binding_hash
        assert core_hash == companion_core_hash
    cases = {}
    graph_cases = {}
    graph_errors = {}
    result = {
        "cases": cases,
        "graph_cases": graph_cases,
        "graph_errors": graph_errors,
        "graph_mode": args.graph_mode,
        "allow_host_build_difference": args.allow_host_build_difference,
        "core_path": str(core),
        "core_embedded_in_binding": core_embedded_in_binding,
        "core_sha256": core_hash,
        "binding_path": str(binding),
        "binding_sha256": binding_hash,
        "companion_core_sha256": companion_core_hash,
        "companion_binding_sha256": companion_binding_hash,
        "plugin_path": str(args.plugin.resolve()),
        "plugin_sha256": hashlib.sha256(args.plugin.read_bytes()).hexdigest(),
        "padding_shortcut_env": os.environ.get("ORT_PACKED_QSA_MERGE_PADDING_SHORTCUT"),
        "mapped_libraries": sorted(str(path) for path in mapped),
    }

    def checkpoint():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    for dtype in (np.float32, np.float16):
        for width in (1, 2, 4):
            for kind in ("random", "ties", "positive_ties"):
                name = f"{dtype.__name__}_{width}_{kind}"
                model, feeds = make_case(width, 65536, [32768], kind, dtype)
                cases[name] = run_case(devices[0], model, feeds, False)
            for label, capacity, lengths in (
                ("tail", 65536, [32771]),
                ("partial_tile", 65539, [32775]),
                ("overflow", 65536, [262143]),
                ("full", 65536, [262140]),
                ("empty", 65536, [0]),
                ("mixed", 65536, [32768, 3]),
            ):
                name = f"{dtype.__name__}_{width}_{label}"
                model, feeds = make_case(width, capacity, lengths, "random", dtype)
                cases[name] = run_case(devices[0], model, feeds, False)
                checkpoint()
        for active in (0, 1):
            name = f"{dtype.__name__}_packed_active{active}"
            model, feeds = make_case(4, 65536, [32771], "random", dtype, packed=True, active=active)
            cases[name] = run_case(devices[0], model, feeds, False)
            checkpoint()
    if args.graph_mode != "skip":
        for dtype in (np.float32, np.float16):
            for active in (0, 1):
                name = f"{dtype.__name__}_packed_graph_active{active}"
                model, feeds = make_case(4, 65536, [32771], "random", dtype, packed=True, active=active)
                try:
                    graph_cases[name] = run_case(devices[0], model, feeds, True)
                except Exception as error:
                    graph_errors[name] = f"{type(error).__name__}: {error}"
                    checkpoint()
                    if args.graph_mode == "strict":
                        raise
                checkpoint()
    checkpoint()
    if args.reference:
        reference = json.loads(args.reference.read_text())
        assert result["core_sha256"] == reference["core_sha256"]
        assert result["binding_sha256"] == reference["binding_sha256"]
        assert result["core_path"] == reference["core_path"]
        assert result["binding_path"] == reference["binding_path"]
        assert result["cases"] == reference["cases"], "Baseline/candidate output mismatch"
        if args.graph_mode == "strict":
            assert result["graph_cases"] == reference.get("graph_cases", {}), "Graph output mismatch"
        elif args.graph_mode == "separate":
            for name in graph_cases.keys() & reference.get("graph_cases", {}).keys():
                if graph_cases[name] != reference["graph_cases"][name]:
                    graph_errors[name] = "Baseline/candidate graph output mismatch"
            checkpoint()
    print(
        f"PASS: {len(cases)} eager cases, six bitwise output hashes per case; "
        f"graph passed={len(graph_cases)}, failed={len(graph_errors)}, mode={args.graph_mode}"
    )


if __name__ == "__main__":
    main()

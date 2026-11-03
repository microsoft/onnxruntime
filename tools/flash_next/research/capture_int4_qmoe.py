import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source, destination, include_gdn=False):
    import onnx

    model = onnx.load(source / "model.onnx", load_external_data=False)
    node = next(n for n in model.graph.node if n.op_type == "QMoE")
    values = {v.name: v for v in (*model.graph.value_info, *model.graph.input, *model.graph.output)}
    names = [node.input[0], node.input[1], node.output[0]]
    gdn_names = {}
    if include_gdn:
        gdn = next(n for n in model.graph.node if n.op_type == "GatedDeltaNet")
        gdn_names = dict(zip(("packed", "decay", "beta", "a_log", "dt_bias", "output", "state"),
                             [gdn.input[i] for i in (0, 4, 5, 7, 8)] + list(gdn.output[:2])))
    selected = [values[name] for name in names]
    selected.extend(values[name] for name in gdn_names.values())
    for value in selected:
        onnx.checker.check_value_info(value)
    initializers = {t.name: t for t in model.graph.initializer}
    tensors = {}
    for index in range(2, 8):
        tensor = initializers[node.input[index]]
        tensors[str(index)] = {
            "name": tensor.name, "shape": list(tensor.dims), "dtype": tensor.data_type,
            "external": {e.key: e.value for e in tensor.external_data},
        }
    if destination.resolve() != source.resolve():
        raise ValueError("Diagnostic graph must share the original external-data directory")
    prefix = "qmoe-gdn" if include_gdn else "qmoe"
    derived = destination / f"model-{prefix}-capture.onnx"
    if derived.exists():
        raise FileExistsError(derived)
    existing = {v.name for v in model.graph.output}
    for value in selected:
        if value.name not in existing:
            model.graph.output.append(value)
    onnx.save(model, derived)
    manifest = {
        "original_graph_sha256": sha256(source / "model.onnx"),
        "derived_graph_sha256": sha256(derived),
        "derived_graph_filename": derived.name,
        "config_sha256": sha256(source / "genai_config.json"),
        "node": node.name, "capture_names": names, "tensors": tensors,
        "gdn_capture_names": gdn_names,
        "top_k": next(a.i for a in node.attribute if a.name == "k"),
        "validation": "New output ValueInfos checked; original external-data directory retained. "
                      "Runtime full validation required.",
        "scope": "Diagnostic extra outputs only; not a throughput benchmark",
    }
    (destination / f"{prefix}-capture-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def capture(model_path, plugin, control, output, manifest_name):
    import numpy as np
    import onnxruntime_genai as og

    manifest = json.loads((model_path / manifest_name).read_text())
    metadata = json.loads(control.read_text())
    flags = metadata["load_experiment_flags"]
    if len(flags) != 13 or any(os.environ.get(name) != "0" for name in flags):
        raise ValueError("Capture requires all 13 recorded experimental flags explicitly disabled")
    if sha256(model_path / manifest["derived_graph_filename"]) != manifest["derived_graph_sha256"]:
        raise ValueError("Derived graph identity changed")
    if sha256(model_path / "genai_config.json") != metadata["config_sha256"]:
        raise ValueError("Configuration differs from the validated control")
    if sha256(plugin) != metadata["plugin_sha256"]:
        raise ValueError("Provider differs from the validated control")
    og.register_execution_provider_library("webgpu", str(plugin))
    mapped = {line.split(maxsplit=5)[-1] for line in Path("/proc/self/maps").read_text().splitlines()}
    for key, prefix in (("genai_native", "libonnxruntime-genai.so"), ("ort_native", "libonnxruntime.so")):
        paths = {Path(p) for p in mapped if Path(p).name.startswith(prefix)}
        if len(paths) != 1 or sha256(next(iter(paths))) != metadata[key]["sha256"]:
            raise ValueError(f"Mapped {key} differs from the validated control")
    config = og.Config(str(model_path))
    config.overlay(json.dumps({"model": {"decoder": {"filename": manifest["derived_graph_filename"]}}}))
    config.set_provider_option("webgpu", "validationMode", "full")
    print("Loading diagnostic graph", flush=True)
    model = og.Model(config)
    print("Diagnostic graph loaded", flush=True)
    tokenizer = og.Tokenizer(model)
    prompt = metadata["samples"][0]["input_tokens"]
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=len(prompt) + 2, min_length=len(prompt) + 2,
                              do_sample=False, chunk_size=16)
    generator = og.Generator(model, params)
    generator.append_tokens(prompt)
    arrays = {}
    tokens = []
    for step in range(2):
        generator.generate_next_token()
        tokens.append(int(generator.get_next_tokens()[0]))
        for label, name in zip(("hidden", "router", "output"), manifest["capture_names"]):
            value = np.asarray(generator.get_output(name))
            if value.dtype != np.float16 or not np.isfinite(value).all():
                raise ValueError(f"Unexpected dtype or nonfinite capture for {name}")
            arrays[f"{label}_{step}"] = value.reshape(-1, value.shape[-1])[-1].copy()
        for label, name in manifest.get("gdn_capture_names", {}).items():
            value = np.asarray(generator.get_output(name))
            if value.dtype not in (np.float16, np.float32) or not np.isfinite(value).all():
                raise ValueError(f"Unexpected dtype or nonfinite GDN capture for {name}")
            arrays[f"gdn_{label}_{step}"] = value.copy()
    expected = metadata["samples"][0]["output_tokens"][:2]
    if tokens != expected:
        raise ValueError(f"Instrumented graph changed first tokens: {tokens} != {expected}")
    np.savez(output, **arrays)
    report = dict(manifest, status="captured", emitted_tokens=tokens,
                  control_tokens=expected, flags=flags, validation_mode="full",
                  provider_sha256=sha256(plugin), capture_sha256=sha256(output))
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def reference(model_path, capture_path, output):
    import numpy as np
    import onnx

    manifest = json.loads((model_path / "qmoe-capture-manifest.json").read_text())
    capture_metadata = json.loads(capture_path.with_suffix(".json").read_text())
    if capture_metadata["capture_sha256"] != sha256(capture_path):
        raise ValueError("Capture identity changed")
    tensors = {}
    for index, spec in manifest["tensors"].items():
        ext = spec["external"]
        dtype = np.dtype("u1" if spec["dtype"] == 2 else "<f2")
        if int(ext["length"]) != int(np.prod(spec["shape"])) * dtype.itemsize:
            raise ValueError(f"External tensor byte count mismatch: {spec['name']}")
        tensors[index] = np.memmap(model_path / ext["location"], mode="r", dtype=dtype,
                                   offset=int(ext["offset"]), shape=tuple(spec["shape"]))
    model = onnx.load(model_path / "model.onnx", load_external_data=False)
    producers = {name: node for node in model.graph.node for name in node.output}
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    router = producers[producers[manifest["capture_names"][1]].input[0]]
    attributes = {a.name: a.i for a in router.attribute}
    if (router.op_type != "MatMulNBits" or router.input[0] != manifest["capture_names"][0] or
            attributes["bits"] != 4 or attributes["block_size"] != 32 or len(router.input) != 3):
        raise ValueError("Unsupported router projection for this independent diagnostic")
    router_tensors = []
    for name, dtype in zip(router.input[1:], ("u1", "<f2")):
        tensor = initializers[name]
        ext = {entry.key: entry.value for entry in tensor.external_data}
        router_tensors.append(np.memmap(
            model_path / ext["location"], mode="r", dtype=dtype, offset=int(ext["offset"]),
            shape=tuple(tensor.dims)))

    def dequantize(weights, scales):
        unpacked = np.empty((weights.shape[0], weights.shape[1] * 2), dtype=np.float32)
        unpacked[:, 0::2] = weights & 15
        unpacked[:, 1::2] = weights >> 4
        unpacked -= 8
        unpacked *= np.repeat(scales.astype(np.float32), 32, axis=1)
        return unpacked.astype(np.float16).astype(np.float64)

    rows = []
    with np.load(capture_path) as data:
        for step in range(2):
            hidden = data[f"hidden_{step}"].astype(np.float64)
            logits = data[f"router_{step}"].astype(np.float64)
            actual = data[f"output_{step}"].astype(np.float64)
            router_expected = (
                dequantize(router_tensors[0].reshape(logits.size, -1), router_tensors[1]) @ hidden
            ).astype(np.float16).astype(np.float64)
            router_difference = logits - router_expected
            experts = np.lexsort((np.arange(logits.size), -logits))[:manifest["top_k"]]
            weights = np.exp(logits[experts] - logits[experts[0]])
            weights = (weights / weights.sum()).astype(np.float16).astype(np.float64)
            expected = np.zeros_like(actual)
            for expert, weight in sorted(zip(experts.tolist(), weights.tolist())):
                first = (dequantize(tensors["2"][expert], tensors["3"][expert]) @ hidden +
                         tensors["4"][expert].astype(np.float64)).astype(np.float16).astype(np.float64)
                gate, linear = first[0::2], first[1::2]
                activated = (gate / (1 + np.exp(-gate)) * linear).astype(np.float16).astype(np.float64)
                down = (dequantize(tensors["5"][expert], tensors["6"][expert]) @ activated +
                        tensors["7"][expert].astype(np.float64)).astype(np.float16).astype(np.float64)
                expected += weight * down
            expected = expected.astype(np.float16).astype(np.float64)
            difference = actual - expected
            rows.append({
                "step": step, "experts": experts.tolist(), "dimensions": int(actual.size),
                "max_abs": float(np.abs(difference).max()),
                "rms_error": float(np.sqrt(np.mean(difference ** 2))),
                "reference_rms": float(np.sqrt(np.mean(expected ** 2))),
                "nonfinite_actual": int((~np.isfinite(actual)).sum()),
                "nonfinite_reference": int((~np.isfinite(expected)).sum()),
                "exact_elements": int((actual == expected).sum()),
                "router_projection": {
                    "dimensions": int(logits.size),
                    "max_abs": float(np.abs(router_difference).max()),
                    "rms_error": float(np.sqrt(np.mean(router_difference ** 2))),
                    "reference_rms": float(np.sqrt(np.mean(router_expected ** 2))),
                    "exact_elements": int((logits == router_expected).sum()),
                    "reference_topk": np.lexsort(
                        (np.arange(logits.size), -router_expected))[:manifest["top_k"]].tolist(),
                },
            })
    report = {
        "status": "characterized_not_quality_qualification",
        "capture_sha256": sha256(capture_path), "rows": rows,
        "reference": "Independent NumPy float64 products; FP16 dequantization and stage boundaries; "
                     "interleaved SwiGLU; k10 normalized routing and both expert biases",
        "caveat": "Does not emulate the precise GPU reduction tree or prove full-model correctness",
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prep = subparsers.add_parser("prepare")
    prep.add_argument("source", type=Path)
    prep.add_argument("destination", type=Path)
    prep.add_argument("--include-gdn", action="store_true")
    cap = subparsers.add_parser("capture")
    cap.add_argument("--model", type=Path, required=True)
    cap.add_argument("--plugin", type=Path, required=True)
    cap.add_argument("--control", type=Path, required=True)
    cap.add_argument("--output", type=Path, required=True)
    cap.add_argument("--manifest", default="qmoe-capture-manifest.json")
    ref = subparsers.add_parser("reference")
    ref.add_argument("--model", type=Path, required=True)
    ref.add_argument("--capture", type=Path, required=True)
    ref.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.source, args.destination, args.include_gdn)
    elif args.command == "capture":
        capture(args.model, args.plugin, args.control, args.output, args.manifest)
    else:
        reference(args.model, args.capture, args.output)

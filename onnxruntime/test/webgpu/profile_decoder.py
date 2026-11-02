# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Capture/benchmark a WebGPU GenAI decoder with resident KV and optional graph replay.

Requires a native WebGPU ORT Python build, numpy and tokenizers. Static KV also
requires onnx for a small upload session (native Python has no WebGPU allocator
factory). This measures greedy decoder inference, not a complete GenAI application.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

import onnxruntime as ort

DTYPES = {
    "tensor(float16)": np.float16,
    "tensor(float)": np.float32,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
}


def session_options(args, profiling=False):
    options = ort.SessionOptions()
    options.enable_profiling = profiling
    options.profile_file_prefix = str(args.output_dir / "webgpu")
    options.log_severity_level = 3
    options.intra_op_num_threads = args.intra_op_threads
    options.add_session_config_entry(
        "ep.webgpuexecutionprovider.maxNumPendingDispatches", str(args.max_pending_dispatches)
    )
    return options


class DeviceUpload:
    """Use WebGPU I/O binding to allocate/update buffers without changing shared Python bindings."""

    def __init__(self, args):
        self.args = args
        self.sessions = {}

    def upload(self, array, target=None):
        import onnx  # noqa: PLC0415

        key = (array.dtype.str, array.ndim)
        if key not in self.sessions:
            dtype = onnx.helper.np_dtype_to_tensor_dtype(array.dtype)
            graph = onnx.helper.make_graph(
                [onnx.helper.make_node("Identity", ["x"], ["y"])],
                "upload",
                [onnx.helper.make_tensor_value_info("x", dtype, [None] * array.ndim)],
                [onnx.helper.make_tensor_value_info("y", dtype, [None] * array.ndim)],
            )
            model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 21)], ir_version=10)
            options = session_options(self.args)
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            self.sessions[key] = ort.InferenceSession(
                model.SerializeToString(),
                options,
                providers=[("WebGpuExecutionProvider", {"dawnBackendType": self.args.backend})],
            )
        session = self.sessions[key]
        binding = session.io_binding()
        binding.bind_cpu_input("x", array)
        if target is None:
            binding.bind_output("y", "webgpu")
        else:
            binding.bind_ortvalue_output("y", target)
        session.run_with_iobinding(binding)
        binding.synchronize_outputs()
        return binding.get_outputs()[0]


def capture(args):
    config = json.loads((args.model_dir / "genai_config.json").read_text(encoding="utf-8"))["model"]
    decoder = config["decoder"]
    model = args.model_dir / decoder["filename"]
    tokenizer = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    prompt = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else args.prompt
    tokens = tokenizer.encode(prompt).ids
    if not tokens:
        raise ValueError("The prompt must produce at least one token.")
    if len(tokens) + args.decode_steps > config["context_length"]:
        raise ValueError("Prompt plus decode steps exceeds the configured context length.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    session = ort.InferenceSession(
        str(model),
        session_options(args, not args.no_profile),
        providers=[
            (
                "WebGpuExecutionProvider",
                {"dawnBackendType": args.backend, "enableGraphCapture": "1" if args.graph_capture else "0"},
            )
        ],
        disabled_optimizers=set(args.disable_optimizer),
    )
    if "WebGpuExecutionProvider" not in session.get_providers():
        raise RuntimeError("This ORT build does not provide WebGPU.")
    input_types = {value.name: value.type for value in session.get_inputs()}
    output_types = {value.name: value.type for value in session.get_outputs()}
    output_names = list(output_types)
    logits_name = decoder["outputs"]["logits"]
    ids_name = decoder["inputs"]["input_ids"]
    mask_name = decoder["inputs"]["attention_mask"]
    pairs = [
        (decoder["inputs"][f"past_{kind}_names"] % layer, decoder["outputs"][f"present_{kind}_names"] % layer)
        for layer in range(decoder["num_hidden_layers"])
        for kind in ("key", "value")
    ]
    capacity = len(tokens) + args.decode_steps if args.static_kv else 0
    kv_shape = (1, decoder["num_key_value_heads"], capacity, decoder["head_size"])
    static_values = {}
    if args.static_kv:
        uploader = DeviceUpload(args)
        for name, _ in pairs:
            # Zero initialization is outside measurement. Stale tail entries on later
            # iterations must remain masked; changing prompts are validated separately.
            static_values[name] = uploader.upload(np.zeros(kv_shape, dtype=DTYPES[input_types[name]]))
        logits_value = uploader.upload(np.zeros([1, 1, config["vocab_size"]], dtype=DTYPES[output_types[logits_name]]))
        masks = np.zeros((1, capacity), dtype=DTYPES[input_types[mask_name]])
        mask_value = uploader.upload(masks)
        id_values = {
            stage: uploader.upload(np.asarray([values], dtype=DTYPES[input_types[ids_name]]))
            for stage, values in (("prefill", tokens), ("decode", [0]))
        }
    bindings = {}
    runs = []
    sequences = []
    generated_ids = []
    validation = {}
    validation_logits = []
    for iteration in range(args.warmup + args.repeat):
        warmup = iteration < args.warmup
        kv = (
            static_values
            if args.static_kv
            else {
                name: ort.OrtValue.ortvalue_from_numpy(np.empty(kv_shape, dtype=DTYPES[input_types[name]]))
                for name, _ in pairs
            }
        )
        current = tokens
        generated = []
        total_length = 0
        for step in range(args.decode_steps + 1):
            step_start = time.perf_counter()
            total_length += len(current)
            stage = "prefill" if step == 0 else "decode"
            reuse = args.graph_capture and stage in bindings
            if not reuse:
                binding = session.io_binding()
                if args.graph_capture:
                    bindings[stage] = binding
            else:
                binding = bindings[stage]
            ids = np.asarray([current], dtype=DTYPES[input_types[ids_name]])
            if args.static_kv:
                uploader.upload(ids, id_values[stage])
                masks.fill(0)
                masks[:, :total_length] = 1
                uploader.upload(masks, mask_value)
                if not reuse:
                    binding.bind_ortvalue_input(ids_name, id_values[stage])
                    binding.bind_ortvalue_input(mask_name, mask_value)
            else:
                binding.bind_cpu_input(ids_name, ids)
                binding.bind_cpu_input(mask_name, np.ones((1, total_length), dtype=DTYPES[input_types[mask_name]]))
            if not reuse:
                for name, value in kv.items():
                    binding.bind_ortvalue_input(name, value)
                if args.static_kv:
                    binding.bind_ortvalue_output(logits_name, logits_value)
                    for name, present in pairs:
                        binding.bind_ortvalue_output(present, kv[name])
                else:
                    for name in output_names:
                        binding.bind_output(name, "cpu" if name == logits_name else "webgpu")
            run_options = ort.RunOptions()
            run_options.add_run_config_entry("gpu_graph_id", str(0 if step == 0 else 1))
            start = time.perf_counter()
            session.run_with_iobinding(binding, run_options)
            binding.synchronize_outputs()
            outputs = dict(zip(output_names, binding.get_outputs(), strict=True))
            logits = outputs[logits_name].numpy()
            elapsed_ms = (time.perf_counter() - start) * 1000
            # Include logits readback in both timing modes. step_wall_ms additionally
            # includes binding/input updates and greedy token selection.
            if not np.isfinite(logits).all():
                raise RuntimeError("Decoder produced non-finite logits.")
            current = [int(np.argmax(logits[0, -1]))]
            generated.extend(current)
            kv = {name: outputs[present] for name, present in pairs}
            step_wall_ms = (time.perf_counter() - step_start) * 1000
            if args.validation_output and iteration == args.warmup:
                validation_logits.append(logits[0, -1].copy())
                if step == args.decode_steps:
                    for _, present in pairs:
                        validation[present] = outputs[present].numpy()[:, :, :total_length, :]
            runs.append(
                {
                    "iteration": iteration,
                    "warmup": warmup,
                    "stage": stage,
                    "step": step,
                    "input_tokens": len(tokens) if step == 0 else 1,
                    "total_tokens": total_length,
                    "wall_ms": elapsed_ms,
                    "step_wall_ms": step_wall_ms,
                }
            )
        sequences.append(tokenizer.decode(generated))
        generated_ids.append(generated)
    trace = session.end_profiling() if not args.no_profile else None
    if args.validation_output:
        validation["logits"] = np.asarray(validation_logits)
        args.validation_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.validation_output, **validation)
    metadata = {
        "model": str(model.resolve()),
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "ort_version": ort.__version__,
        "ort_module": ort.__file__,
        "backend": args.backend,
        "static_kv": args.static_kv,
        "graph_capture": args.graph_capture,
        "disabled_optimizers": args.disable_optimizer,
        "max_pending_dispatches": args.max_pending_dispatches,
        "intra_op_threads": args.intra_op_threads,
        "command": sys.argv,
        "prompt_tokens": len(tokens),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "decode_steps": args.decode_steps,
        "repeat": args.repeat,
        "warmup": args.warmup,
        "trace": trace,
        "runs": runs,
        "generated_text": sequences,
        "generated_token_ids": generated_ids,
        "workload": "Batch 1 greedy decoder; resident KV; logits readback; fixed steps ignoring EOS",
        "timing": "wall_ms: run, synchronize, outputs and CPU logits; step_wall_ms: also input uploads, binding, greedy selection",
    }
    (args.output_dir / "capture.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    # Release replay references before upload sessions and their buffer owners are destroyed.
    if args.graph_capture:
        for graph_id in (0, 1):
            session.release_captured_graph(graph_id)
    print(
        json.dumps(
            {
                "trace": trace,
                "metadata": str(args.output_dir / "capture.json"),
                "prompt_tokens": len(tokens),
                "runs": len(runs),
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prompt", default="Explain how a GPU executes a neural network and where CPU overhead can arise."
    )
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--backend", choices=["D3D12", "Vulkan", "Metal"], default="D3D12")
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument("--max-pending-dispatches", type=int, default=16)
    parser.add_argument("--intra-op-threads", type=int, default=0)
    parser.add_argument(
        "--static-kv", action="store_true", help="Preallocate and reuse GPU KV/logits buffers; requires onnx."
    )
    parser.add_argument(
        "--graph-capture", action="store_true", help="Replay fixed prefill/decode bindings; requires static-kv."
    )
    parser.add_argument(
        "--disable-optimizer",
        action="append",
        default=[],
        help="Disable a named ORT optimizer for isolation experiments.",
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        help="Save all-step logits and final live KV from the first measured iteration.",
    )
    args = parser.parse_args()
    if args.graph_capture and not args.static_kv:
        parser.error("graph-capture requires static-kv")
    if args.decode_steps < 0 or args.repeat < 1 or args.warmup < 0:
        parser.error("decode-steps/warmup must be nonnegative and repeat must be positive")
    if not 1 <= args.max_pending_dispatches <= 2048 or args.intra_op_threads < 0:
        parser.error("max-pending-dispatches must be in [1, 2048] and intra-op-threads must be nonnegative")
    capture(args)


if __name__ == "__main__":
    main()

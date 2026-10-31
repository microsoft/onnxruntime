# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Capture a WebGPU GenAI decoder's prefill/decode trace with GPU-resident KV outputs.

Requires a native WebGPU ORT Python build, numpy and tokenizers. This invokes ORT
directly; it measures greedy decoder inference, not the complete GenAI application.
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
    options = ort.SessionOptions()
    options.enable_profiling = not args.no_profile
    options.profile_file_prefix = str(args.output_dir / "webgpu")
    options.log_severity_level = 3
    session = ort.InferenceSession(
        str(model),
        options,
        providers=[("WebGpuExecutionProvider", {"dawnBackendType": args.backend, "enableGraphCapture": "0"})],
    )
    if "WebGpuExecutionProvider" not in session.get_providers():
        raise RuntimeError("This ORT build does not provide WebGPU.")
    input_types = {value.name: value.type for value in session.get_inputs()}
    output_names = [value.name for value in session.get_outputs()]
    logits_name = decoder["outputs"]["logits"]
    pairs = []
    for layer in range(decoder["num_hidden_layers"]):
        for kind in ("key", "value"):
            pairs.append(
                (decoder["inputs"][f"past_{kind}_names"] % layer, decoder["outputs"][f"present_{kind}_names"] % layer)
            )
    kv_shape = (1, decoder["num_key_value_heads"], 0, decoder["head_size"])
    runs = []
    sequences = []
    for iteration in range(args.warmup + args.repeat):
        warmup = iteration < args.warmup
        kv = {
            name: ort.OrtValue.ortvalue_from_numpy(
                np.empty(
                    kv_shape, dtype={"tensor(float16)": np.float16, "tensor(float)": np.float32}[input_types[name]]
                )
            )
            for name, _ in pairs
        }
        current = tokens
        generated = []
        total_length = 0
        for step in range(args.decode_steps + 1):
            total_length += len(current)
            binding = session.io_binding()
            for role, array in (
                ("input_ids", np.asarray([current])),
                ("attention_mask", np.ones((1, total_length))),
            ):
                name = decoder["inputs"][role]
                dtype = {"tensor(int64)": np.int64, "tensor(int32)": np.int32}[input_types[name]]
                binding.bind_cpu_input(name, array.astype(dtype))
            for name, value in kv.items():
                binding.bind_ortvalue_input(name, value)
            for name in output_names:
                binding.bind_output(name, "cpu" if name == logits_name else "webgpu")
            start = time.perf_counter()
            session.run_with_iobinding(binding)
            binding.synchronize_outputs()
            elapsed_ms = (time.perf_counter() - start) * 1000
            outputs = dict(zip(output_names, binding.get_outputs(), strict=True))
            logits = outputs[logits_name].numpy()
            if not np.isfinite(logits).all():
                raise RuntimeError("Decoder produced non-finite logits.")
            current = [int(np.argmax(logits[0, -1]))]
            generated.extend(current)
            kv = {name: outputs[present] for name, present in pairs}
            runs.append(
                {
                    "iteration": iteration,
                    "warmup": warmup,
                    "stage": "prefill" if step == 0 else "decode",
                    "step": step,
                    "input_tokens": len(tokens) if step == 0 else 1,
                    "total_tokens": total_length,
                    "wall_ms": elapsed_ms,
                }
            )
        sequences.append(tokenizer.decode(generated))
    trace = session.end_profiling() if not args.no_profile else None
    metadata = {
        "model": str(model.resolve()),
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "ort_version": ort.__version__,
        "ort_module": ort.__file__,
        "backend": args.backend,
        "command": sys.argv,
        "prompt_tokens": len(tokens),
        "decode_steps": args.decode_steps,
        "repeat": args.repeat,
        "warmup": args.warmup,
        "trace": trace,
        "runs": runs,
        "generated_text": sequences,
        "workload": "Batch 1 greedy decoder; graph capture disabled; GPU-resident KV; CPU logits; fixed steps ignoring EOS",
    }
    (args.output_dir / "capture.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
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
    parser = argparse.ArgumentParser(description=__doc__)
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
    args = parser.parse_args()
    if args.decode_steps < 0 or args.repeat < 1 or args.warmup < 0:
        parser.error("decode-steps/warmup must be nonnegative and repeat must be positive")
    capture(args)


if __name__ == "__main__":
    main()

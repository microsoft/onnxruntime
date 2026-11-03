import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np


def summarize(samples):
    return {
        "e2e_tps": sum(s["output_tokens"] for s in samples) / sum(s["request_seconds"] for s in samples),
        "decode_tps": sum(s["output_tokens"] - 1 for s in samples) / sum(s["decode_seconds"] for s in samples),
        "median_decode_tps": float(np.median([s["decode_tps"] for s in samples])),
        "mean_ttft_seconds": sum(s["ttft_seconds"] for s in samples) / len(samples),
    }


def token_hash(tokens):
    return hashlib.sha256(np.asarray(tokens, dtype="<i4").tobytes()).hexdigest()


def natural_text_prompt(tokens, length, seed):
    if len(tokens) < length:
        raise ValueError(f"Text pool has {len(tokens)} tokens; {length} are required")
    offset = (seed * 7919) % max(len(tokens) - length, 1)
    return np.asarray(tokens[offset:offset + length], dtype=np.int32), offset


def main():
    import onnxruntime_genai as og

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--width", type=int, default=7)
    parser.add_argument("--expert-quantization", choices=("fp8", "nvfp4"), default="fp8")
    parser.add_argument("--profile-prefix", help="Enable ORT session profiling; timings are diagnostic only")
    parser.add_argument("--plugin", default=(
        "/workspace/onnxruntime-webgpu-fp8-checkpoint/build-webgpu-fp8/Release/"
        "libonnxruntime_providers_webgpu.so"
    ))
    parser.add_argument("--compare-experiments", action="store_true",
                        help="Counterbalance candidate flags off/on in one loaded model; use the candidate plugin")
    parser.add_argument("--context-tokens", type=int, help="Exact natural-text input length, without chat wrapping")
    parser.add_argument("--text-file", type=Path, help="Frozen natural-text pool, shared with the reference benchmark")
    parser.add_argument("--prompt-pool-manifest", type=Path, help="Require frozen text, tokenizer and token-pool hashes")
    parser.add_argument("--cache-blocks", type=int, help="Explicit cache budget for an isolated dynamic-batching config")
    parser.add_argument("--prefill-chunk", type=int, help="Explicit Engine search.chunk_size; record for comparison")
    args = parser.parse_args()
    if args.tokens < 2 or args.repetitions < 1:
        parser.error("At least two output tokens and one measured repetition are required")
    if args.width < 0:
        parser.error("--width must be nonnegative")
    if args.compare_experiments and args.profile_prefix:
        parser.error("Use unprofiled requests for the experimental comparison")
    if (args.context_tokens is None) != (args.text_file is None):
        parser.error("--context-tokens and --text-file must be supplied together")
    if args.context_tokens is not None and args.context_tokens < 1:
        parser.error("--context-tokens must be positive")
    if args.cache_blocks is not None and args.cache_blocks < 1:
        parser.error("--cache-blocks must be positive")
    if args.prefill_chunk is not None and args.prefill_chunk < 1:
        parser.error("--prefill-chunk must be positive")
    if args.text_file is not None and not args.text_file.is_file():
        parser.error(f"Text pool does not exist: {args.text_file}")
    if args.prompt_pool_manifest is not None and args.text_file is None:
        parser.error("--prompt-pool-manifest requires --text-file")
    output = Path(args.output)
    result = {
        "status": "running", "model": args.model, "provider": "WebGPU",
        "configured_draft_width": args.width, "samples": [],
        "expert_quantization": args.expert_quantization, "quality_qualified": False,
        "policy": ("one excluded warmup per mode; counterbalanced pairs; fresh request per sample"
                   if args.compare_experiments else
                   "one excluded warmup; fresh request per sample; all measured samples retained"),
        "profiling_enabled": args.profile_prefix is not None,
        "plugin": args.plugin,
        "experimental_flags": {name: os.environ.get(name, "0") for name in (
            "ORT_WEBGPU_QMOE_FP8_COOPERATIVE", "ORT_WEBGPU_QMOE_SMALL_BATCH",
            "ORT_WEBGPU_QMOE_NVFP4_PREFILL",
        )},
        "decode_metric": "(output_tokens-1)/(last_token_time-first_token_time)",
        "timing_policy": "Start before begin_turn; one delivery timestamp per Engine.run, shared by its token events",
        "workload": ("fixed-natural-text-context" if args.context_tokens is not None
                     else "45-token-compatibility-diagnostic"),
        "requested_context_tokens": args.context_tokens,
        "requested_new_tokens": args.tokens,
        "limitations": [
            "cache capacity and graph-capture settings follow recorded config and overlays",
            "CPU Engram, host n-gram metadata validation and host compact-state replay",
            ("software E2M1/E4M3FN weight decoding, not native FP4 matrix instructions"
             if args.expert_quantization == "nvfp4" else
             "software E4M3FN weight decoding, not native FP8 matrix instructions"),
            "context matching alone does not establish matching export, quantization or runtime",
        ],
    }
    try:
        if args.prompt_pool_manifest is not None:
            manifest = json.loads(args.prompt_pool_manifest.read_text())
            if hashlib.sha256(args.text_file.read_bytes()).hexdigest() != manifest["pool_sha256"]:
                raise ValueError("Natural-text pool does not match the frozen manifest")
            tokenizer_path = Path(args.model) / "tokenizer.json"
            if hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() != manifest["tokenizer_sha256"]:
                raise ValueError("Tokenizer does not match the frozen manifest")
            result["prompt_pool_manifest"] = manifest
        og.set_log_options(enabled=True, warning=True, ansi_tags=False)
        og.register_execution_provider_library("webgpu", args.plugin)
        config = og.Config(args.model)
        config_path = Path(args.model) / "genai_config.json"
        source_config = json.loads(config_path.read_text())
        result["source_config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
        result["source_engine_config"] = source_config.get("engine")
        result["source_decoder_session_options"] = source_config["model"]["decoder"]["session_options"]
        result["plugin_sha256"] = hashlib.sha256(Path(args.plugin).read_bytes()).hexdigest()
        result["model_overlay_manifests"] = {
            name: json.loads((Path(args.model) / name).read_text())
            for name in ("webgpu-overlay.json", "diagnostic-policy.json")
            if (Path(args.model) / name).is_file()
        }
        overlay = {
            "model": {
                name: {"session_options": {"log_severity_level": 3}}
                for name in ("decoder", "mtp")
            },
        }
        if args.profile_prefix:
            for name in ("decoder", "mtp"):
                overlay["model"][name]["session_options"]["enable_profiling"] = (
                    args.profile_prefix + "-" + name
                )
        overlay["model"]["mtp"]["enabled"] = args.width != 0
        if args.width != 0:
            overlay["speculative"] = {"max_draft_tokens": args.width}
        if args.cache_blocks is not None:
            batching = source_config.get("engine", {}).get("dynamic_batching")
            if batching is None or "gpu_utilization_factor" in batching:
                raise ValueError("--cache-blocks requires an isolated explicit-block dynamic-batching config")
            overlay["engine"] = {"dynamic_batching": {"num_blocks": args.cache_blocks}}
        if args.prefill_chunk is not None:
            overlay["search"] = {"chunk_size": args.prefill_chunk}
        batching = overlay.get("engine", source_config.get("engine", {})).get("dynamic_batching", {})
        if args.cache_blocks is None:
            batching = source_config.get("engine", {}).get("dynamic_batching", {})
        configured_blocks = batching.get("num_blocks")
        block_size = source_config.get("engine", {}).get("dynamic_batching", {}).get("block_size")
        if (args.context_tokens is not None and configured_blocks is not None and block_size is not None
                and args.context_tokens + args.tokens > configured_blocks * block_size + 1):
            raise ValueError("Requested context plus generation exceeds the configured cache even before auxiliary reserves")
        config.overlay(json.dumps(overlay))
        result["config_overlay"] = overlay
        print(json.dumps({"stage": "model_load", "model": args.model,
                          "expert_quantization": args.expert_quantization}), flush=True)
        started = time.perf_counter()
        model = og.Model(config)
        result["load_seconds"] = time.perf_counter() - started
        print(json.dumps({"stage": "model_loaded", "load_seconds": result["load_seconds"]}), flush=True)
        engine = og.Engine(model)
        tokenizer = og.Tokenizer(model)
        prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\nList the integers from 1 to 50 in order, separated by commas. "
            "Do not add commentary.<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        diagnostic_ids = np.asarray(tokenizer.encode(prompt), dtype=np.int32)
        text_tokens = None
        if args.text_file is not None:
            text_bytes = args.text_file.read_bytes()
            text_tokens = tokenizer.encode(text_bytes.decode("utf-8", errors="replace"))
            result["text_file_sha256"] = hashlib.sha256(text_bytes).hexdigest()
            result["text_tokens_sha256"] = token_hash(text_tokens)
            result["text_pool_tokens"] = len(text_tokens)
            if args.prompt_pool_manifest is not None and (
                result["text_tokens_sha256"] != manifest["tokens_sha256"]
                or len(text_tokens) != manifest["pool_tokens"]
            ):
                raise ValueError("GenAI tokenization differs from the frozen CPU reference pool")
            seeds = [1000 + args.context_tokens] + [i * 7 + args.context_tokens for i in range(args.repetitions)]
            hashes = {
                token_hash(natural_text_prompt(text_tokens, args.context_tokens, seed)[0]) for seed in seeds
            }
            if len(hashes) != len(seeds):
                raise ValueError("Text pool/seed selection repeats prompts; distinct warmup and measured prompts are required")
        result["prompt_tokens"] = args.context_tokens if text_tokens is not None else len(diagnostic_ids)
        result["max_request_length"] = engine.get_capabilities().max_request_length
        required_length = result["prompt_tokens"] + args.tokens
        if required_length > result["max_request_length"]:
            raise ValueError(
                f"Requested {required_length} total tokens exceeds Engine capacity "
                f"{result['max_request_length']}; no request was submitted"
            )
        if args.compare_experiments:
            sequence = [("control", True, -1), ("candidate", True, -1)]
            for repetition in range(args.repetitions):
                modes = ("control", "candidate") if repetition % 2 == 0 else ("candidate", "control")
                sequence.extend((mode, False, repetition) for mode in modes)
        else:
            sequence = [("configured", index == 0, index - 1) for index in range(args.repetitions + 1)]
        for index, (mode, warmup, pair_index) in enumerate(sequence):
            seed = None
            offset = None
            if text_tokens is not None:
                seed = (1000 + args.context_tokens) if warmup else pair_index * 7 + args.context_tokens
                input_ids, offset = natural_text_prompt(text_tokens, args.context_tokens, seed)
            else:
                input_ids = diagnostic_ids
            if args.compare_experiments:
                for name in result["experimental_flags"]:
                    os.environ[name] = "1" if mode == "candidate" else "0"
            before = engine.get_speculative_stats()
            request = engine.create_request()
            options = og.TurnOptions(request)
            options.set_do_sample(False)
            options.set_min_generated_tokens(args.tokens)
            options.set_max_generated_tokens(args.tokens)
            events = engine.create_event_buffer(16)
            generated = []
            first = None
            last = None
            started = time.perf_counter()
            request.begin_turn(input_ids, options)
            while engine.has_pending_requests():
                emitted = engine.run(events)
                delivered = time.perf_counter()
                for event in emitted:
                    if event.flags & (og.EngineEventFlags.FAILED | og.EngineEventFlags.CAPACITY_BLOCKED):
                        raise RuntimeError(f"Engine request failed or capacity blocked: {event.flags}")
                    if int(event.error_code) != 0:
                        raise RuntimeError(f"Engine event failed: {event.error_code}")
                    if event.token is not None:
                        last = delivered
                        if first is None:
                            first = last
                        generated.append(event.token)
            finished = time.perf_counter()
            after = engine.get_speculative_stats()
            request.close()
            stats = {key: after[key] - before[key] for key in (
                "completed_rounds", "mtp_failures", "standard_fallback_steps",
                "draft_tokens_proposed", "draft_tokens_evaluated", "draft_tokens_accepted",
                "draft_forward_passes", "target_forward_passes",
            )}
            if first is None or last is None or len(generated) != args.tokens:
                raise RuntimeError(f"Expected {args.tokens} output tokens; observed {len(generated)}")
            if last <= first:
                raise RuntimeError("Token timing interval must be positive")
            sample = {
                "index": index, "mode": mode, "warmup": warmup, "output_tokens": len(generated),
                "pair_index": pair_index, "prompt_tokens": len(input_ids), "prompt_seed": seed,
                "prompt_offset": offset, "prompt_sha256": token_hash(input_ids),
                "generated_tokens": generated, "output_text": tokenizer.decode(generated),
                "ttft_seconds": first - started, "request_seconds": finished - started,
                "decode_seconds": last - first,
                "decode_to_finish_seconds": finished - first, "speculative_stats": stats,
                "decode_tps": (len(generated) - 1) / (last - first),
                "decode_ms_per_cycle": ((last - first) * 1000 / stats["completed_rounds"]
                                       if stats["completed_rounds"] else None),
                "draft_acceptance_fraction": (stats["draft_tokens_accepted"] / stats["draft_tokens_evaluated"]
                                             if stats["draft_tokens_evaluated"] else None),
            }
            if warmup:
                if args.compare_experiments:
                    result.setdefault("warmups", []).append(sample)
                else:
                    result["warmup"] = sample
            else:
                result["samples"].append(sample)
            output.write_text(json.dumps(result, indent=2))
            print(json.dumps(sample), flush=True)
            if args.width > 0 and (stats["mtp_failures"] or not stats["completed_rounds"]):
                raise RuntimeError("MTP failed or did not complete any speculative round")
            if args.width == 0 and (stats["completed_rounds"] or stats["draft_forward_passes"]):
                raise RuntimeError("Decoder control unexpectedly executed MTP")
        samples = result["samples"]
        if args.compare_experiments:
            result["by_mode"] = {
                mode: summarize([sample for sample in samples if sample["mode"] == mode])
                for mode in ("control", "candidate")
            }
            parity = []
            for pair_index in range(args.repetitions):
                pair = {sample["mode"]: sample for sample in samples if sample["pair_index"] == pair_index}
                if pair["control"]["prompt_sha256"] != pair["candidate"]["prompt_sha256"]:
                    raise RuntimeError("Control and candidate received different prompts")
                parity.append(pair["control"]["generated_tokens"] == pair["candidate"]["generated_tokens"])
            if text_tokens is None:
                for mode in ("control", "candidate"):
                    if len({tuple(sample["generated_tokens"]) for sample in samples if sample["mode"] == mode}) != 1:
                        raise RuntimeError("Repeated greedy requests within a mode generated different tokens")
            result["token_parity_by_pair"] = parity
            result["tokens_match_control"] = all(parity)
            result["quality_qualified"] = False
        else:
            result.update(summarize(samples))
        result["status"] = "completed"
        output.write_text(json.dumps(result, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k not in ("samples", "warmup")}), flush=True)
    except (RuntimeError, ValueError) as error:
        result.update(status="failed", error=str(error))
        output.write_text(json.dumps(result, indent=2))
        raise


if __name__ == "__main__":
    main()

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time


CASES = [
    ("What is 2 + 2? Reply with only the number.", "4"),
    ("What is the capital of France? Reply with only the city name.", "Paris"),
    ("Repeat exactly this word and nothing else: blueberry", "blueberry"),
]


def render_chat(template, content):
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise ValueError(message)

    environment = ImmutableSandboxedEnvironment(autoescape=False)
    environment.globals["raise_exception"] = raise_exception
    return environment.from_string(template).render(
        messages=[{"role": "user", "content": content}],
        enable_thinking=False,
        add_generation_prompt=True,
    )


def run_case(og, model, tokenizer, prompt, chunk, eos_tokens):
    tokens = [int(token) for token in tokenizer.encode(prompt)]
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=len(tokens) + 32, do_sample=False, chunk_size=chunk)
    generator = og.Generator(model, params)
    start = time.perf_counter()
    generator.append_tokens(tokens)
    output = []
    stopped_on_eos = False
    while not generator.is_done():
        generator.generate_next_token()
        token = int(generator.get_next_tokens()[0])
        output.append(token)
        if token in eos_tokens:
            stopped_on_eos = True
            break
    text_tokens = output[:-1] if stopped_on_eos else output
    sequence = list(generator.get_sequence(0))
    # CPU greedy search reports EOS but does not append it to the sequence.
    if sequence != list(tokens) + text_tokens:
        raise RuntimeError(
            f"Generator sequence mismatch: prompt={tokens}, emitted={output}, sequence={sequence}"
        )
    result = {
        "input_tokens": tokens,
        "output_tokens": output,
        "output_text": tokenizer.decode(text_tokens),
        "stopped_on_eos": stopped_on_eos,
        "request_seconds": time.perf_counter() - start,
    }
    del generator
    gc.collect()
    return result


def run_suite(og, model, tokenizer, template, eos_tokens, on_sample):
    for chat, chunk in ((False, 16), (True, 16), (True, 1)):
        for content, expected in CASES:
            prompt = render_chat(template, content) if chat else content
            sample = run_case(og, model, tokenizer, prompt, chunk, eos_tokens)
            sample.update(
                chat_template_applied=chat, chunk=chunk, content=content, expected=expected,
                exact_answer_passed=sample["output_text"].strip() == expected,
            )
            on_sample(sample)


def main():
    import onnxruntime_genai as og

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/workspace/qwen38-int4-webgpu-perf")
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--provider", choices=["webgpu", "cuda"], default="webgpu")
    parser.add_argument("--fp32-accumulation", action="store_true")
    parser.add_argument("--profile-prefix")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.fp32_accumulation and args.provider != "webgpu":
        parser.error("--fp32-accumulation requires the WebGPU provider")
    root = Path(args.model)
    template = (root / "chat_template.jinja").read_text()
    config_data = json.loads((root / "genai_config.json").read_text())
    eos_tokens = config_data["model"]["eos_token_id"]
    if isinstance(eos_tokens, int):
        eos_tokens = [eos_tokens]
    result = {
        "status": "running",
        "scope": "Three short deterministic sanity prompts, not a model-quality benchmark",
        "model": args.model,
        "provider": args.provider,
        "fp32_accumulation": args.fp32_accumulation,
        "profiling": bool(args.profile_prefix),
        "experimental_flags": {name: value for name, value in os.environ.items()
                               if name.startswith("ORT_WEBGPU_")},
        "plugin_sha256": hashlib.sha256(Path(args.plugin).read_bytes()).hexdigest(),
        "graph_sha256": hashlib.sha256((root / "model.onnx").read_bytes()).hexdigest(),
        "chat_template_sha256": hashlib.sha256((root / "chat_template.jinja").read_bytes()).hexdigest(),
        "samples": [],
    }

    def persist():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    persist()
    try:
        if args.provider == "webgpu":
            og.register_execution_provider_library("webgpu", args.plugin)
        config = og.Config(args.model)
        if args.fp32_accumulation:
            config.set_provider_option("webgpu", "enableMatmulFp32Accumulation", "1")
        if args.profile_prefix:
            config.overlay(json.dumps({"model": {"decoder": {
                "session_options": {"enable_profiling": args.profile_prefix},
            }}}))
        if args.provider == "cuda":
            config.clear_providers()
            config.append_provider("cuda")
            config.set_provider_option("cuda", "enable_cuda_graph", "0")
            config.set_provider_option("cuda", "use_tf32", "0")
        print("Loading model", flush=True)
        start = time.perf_counter()
        model = og.Model(config)
        result["model_load_seconds"] = time.perf_counter() - start
        print("Model loaded", flush=True)
        tokenizer = og.Tokenizer(model)
        mapped_paths = {line.split(maxsplit=5)[-1]
                        for line in Path("/proc/self/maps").read_text().splitlines()}
        for label, predicate in (
            ("genai_native", lambda path: path.endswith("/libonnxruntime-genai.so")),
            ("ort_native", lambda path: path.rsplit("/", 1)[-1].startswith("libonnxruntime.so")),
        ):
            paths = [path for path in mapped_paths if predicate(path)]
            if len(paths) != 1:
                raise RuntimeError(f"Expected one mapped {label} library, found {paths}")
            path = paths[0]
            result[label] = {"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        if args.provider == "cuda":
            paths = [path for path in mapped_paths if path.endswith("/libonnxruntime_providers_cuda.so")]
            if len(paths) != 1:
                raise RuntimeError(f"Expected one mapped CUDA provider, found {paths}")
            actual_sha = hashlib.sha256(Path(paths[0]).read_bytes()).hexdigest()
            if actual_sha != result["plugin_sha256"]:
                raise RuntimeError(f"Loaded CUDA provider differs from requested library: {paths[0]}")
            result["cuda_provider_native"] = {"path": paths[0], "sha256": actual_sha}
        persist()
        def on_sample(sample):
            result["samples"].append(sample)
            print(json.dumps(sample), flush=True)
            persist()

        run_suite(og, model, tokenizer, template, eos_tokens, on_sample)
        result["status"] = "completed"
        persist()
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        persist()
        raise


if __name__ == "__main__":
    main()

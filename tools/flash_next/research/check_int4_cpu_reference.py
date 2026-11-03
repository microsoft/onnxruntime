import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from check_int4_chat_quality import CASES, render_chat, run_case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if Path("/dev/nvidiactl").exists() or Path("/dev/dri").exists():
        raise RuntimeError("The CPU reference must run without GPU device access")
    result = {
        "status": "starting",
        "provider": "cpu",
        "scope": "Three wrapped chunk16 sanity prompts; not a throughput or model-quality benchmark",
        "gpu_access": False,
        "graph_sha256": hashlib.sha256((args.model / "model.onnx").read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256((args.model / "genai_config.json").read_bytes()).hexdigest(),
        "cpu_threads": 16,
        "qmoe_cpu_qnbit_gemm_env": os.environ.get("ORT_QMOE_CPU_QNBIT_GEMM"),
        "samples": [],
    }

    def persist():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    persist()
    import onnxruntime_genai as og

    try:
        mapped = {line.split(maxsplit=5)[-1] for line in Path("/proc/self/maps").read_text().splitlines()}
        for label, suffix, expected in (
            ("genai_native", "/libonnxruntime-genai.so",
             "0697af07edc986545f6a54c3649be02af116aba616dc71292cd46ae121a7faa9"),
            ("ort_native", "/libonnxruntime.so.1.32.0",
             "7a4b7503353a7a5fc5256727f4f06ede3b2807cb1a15c7ec530d1b5746ed7f95"),
        ):
            paths = [Path(path) for path in mapped if path.endswith(suffix)]
            if len(paths) != 1:
                raise RuntimeError(f"Expected one mapped {label}, found {paths}")
            actual = hashlib.sha256(paths[0].read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError(f"Unqualified {label} identity: {actual}")
            result[label] = {"path": str(paths[0]), "sha256": actual}
        config = og.Config(str(args.model))
        config.clear_providers()
        config.overlay(json.dumps({"model": {"decoder": {"session_options": {
            "intra_op_num_threads": 16, "inter_op_num_threads": 1,
        }}}}))
        result["status"] = "loading_model"
        persist()
        print(json.dumps({"stage": result["status"]}), flush=True)
        started = time.perf_counter()
        model = og.Model(config)
        result["load_seconds"] = time.perf_counter() - started
        result["status"] = "running_cpu_sanity"
        persist()
        print(json.dumps({"stage": result["status"], "load_seconds": result["load_seconds"]}), flush=True)
        tokenizer = og.Tokenizer(model)
        template = (args.model / "chat_template.jinja").read_text()
        eos = json.loads((args.model / "genai_config.json").read_text())["model"]["eos_token_id"]
        if isinstance(eos, int):
            eos = [eos]
        for content, expected in CASES:
            sample = run_case(og, model, tokenizer, render_chat(template, content), 16, eos)
            sample.update(content=content, expected=expected, chunk=16, chat_template_applied=True,
                          exact_answer_passed=sample["output_text"].strip() == expected)
            result["samples"].append(sample)
            persist()
            print(json.dumps(sample), flush=True)
        result["status"] = "completed"
        persist()
    except RuntimeError as error:
        result.update(status="failed", error=str(error))
        persist()
        raise


if __name__ == "__main__":
    main()

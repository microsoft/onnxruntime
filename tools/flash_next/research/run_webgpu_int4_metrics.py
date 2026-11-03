import argparse
import copy
import json
from pathlib import Path
import subprocess

from build_cpu_greedy_argmax_lanes8 import FLAG as CPU_FLAG
from build_dense_columns8_double_buffer import FLAG as GPU_FLAG
from build_qmoe_node_replay import sha
from benchmark_webgpu_int4_metrics import summarize
from model_cache_preflight import prepare_model_cache, snapshot_model_cache
from prepare_webgpu_int4_metrics import BENCHMARK, GUARD, ROOT
from report_webgpu_int4_metrics import metrics, table
from run_dense_columns8_double_buffer_cpu_local8_standard import RUNTIME, model_command, verify_workers
from run_qmoe_prefill_rows4_standard import TEMPLATE, verify_result


PROVIDER = Path("/home/sayanshaw/repos/qwen38-int4-dense-columns8-double-buffer/build-v1/libonnxruntime_providers_webgpu.so")
REQUESTS = 10


def reference_subset(raw, context):
    subset = copy.deepcopy(raw)
    subset["samples"] = [sample for sample in raw["samples"] if sample["repetition"] < 3]
    subset["summary"] = {f"{context}:configured": summarize(
        [sample for sample in subset["samples"] if not sample["warmup"]])}
    return subset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-completed", action="store_true",
                        help="Revalidate and reuse completed raw runs; never rerun or overwrite existing runs")
    args = parser.parse_args()
    report_path = ROOT / "webgpu-int4-full-metrics-10requests-20261103.json"
    if report_path.exists() and not args.resume_completed:
        raise FileExistsError(report_path)
    prerequisite = ROOT / "dense-columns8-double-buffer-cpu-local8-v1-all-contexts-20261101.qualification.json"
    qualified = json.loads(prerequisite.read_text())
    assert qualified["status"] == "repeated_columns8_double_buffer_cpu_local8_supported_all_contexts"
    assert sha(PROVIDER) == qualified["provider_sha256"]
    assert sha(RUNTIME / "libonnxruntime-genai.so") == qualified["runtime_sha256"]
    for name, digest in qualified["source_sha256"].items():
        assert sha(Path(name)) == digest, name
    parent_path = ROOT / "qmoe-prefill-rows4-v2-contexts-20261031.qualification.json"
    parent = json.loads(parent_path.read_text())
    controls_path = ROOT / "int4-pending64-standard-contexts.json"
    controls = json.loads(controls_path.read_text())
    template_path = ROOT / f"{TEMPLATE}.guard.json"
    template = json.loads(template_path.read_text())["command"]
    model = Path("/home/sayanshaw/repos/qwen38-int4-mlp-input-projections/model")
    assert sha(model / "model.onnx") == parent["graph_sha256"]
    assert sha(model / "genai_config.json") == parent["config_sha256"]
    rows, evidence = [], [Path(__file__), BENCHMARK, GUARD, ROOT / "prepare_webgpu_int4_metrics.py",
                         ROOT / "report_webgpu_int4_metrics.py", prerequisite, parent_path,
                         controls_path, template_path, PROVIDER, RUNTIME / "libonnxruntime-genai.so",
                         ROOT / "benchmark_webgpu_int4.py", ROOT / "shared_activation_guard.py",
                         ROOT / "nvml_monitor.py", model / "model.onnx", model / "genai_config.json"]
    for context in (2048, 8192, 32768):
        prefix = ROOT / f"webgpu-int4-full-metrics-context{context}-10requests-20261103"
        paths = [prefix.with_suffix(suffix) for suffix in (".json", ".guard.json", ".log")]
        paths.append(Path(str(prefix) + "-cache.json"))
        reuse = args.resume_completed and all(path.exists() for path in paths)
        if not reuse and any(path.exists() for path in paths):
            raise FileExistsError(f"Existing incomplete or unapproved run: {prefix}")
        command = model_command(template, PROVIDER, prefix, context, True, "columns8")
        old_benchmark = "/evidence/webgpu-perf-oct20/benchmark_webgpu_int4.py"
        assert command[-1].count(old_benchmark) == 1 and command[-1].count("--repetitions 3") == 1
        command[-1] = command[-1].replace(old_benchmark, "/evidence/webgpu-perf-oct20/" + BENCHMARK.name)
        command[-1] = command[-1].replace("--repetitions 3", f"--repetitions {REQUESTS}")
        command[-1] = "export ORT_GENAI_CPU_GREEDY_ARGMAX_AVX512=0; " + command[-1]
        if not reuse:
            cache = prepare_model_cache(model, prefix)
            subprocess.run(["free", "-h"], check=True)
            subprocess.run(["df", "-h", str(model)], check=True)
            subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.used,utilization.gpu",
                            "--format=csv,noheader"], check=True)
            subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                            "--format=csv,noheader"], check=True)
            subprocess.run(["python3", str(GUARD), "--gpu-index", "7", "--idle-seconds", "3",
                            "--wait-seconds", "15", "--run-seconds", "2400",
                            "--output-prefix", str(prefix), "--", *command], check=True)
            cache["after_run"] = snapshot_model_cache(model)
            Path(cache["evidence_path"]).write_text(json.dumps(cache, indent=2) + "\n")
        else:
            print(f"Revalidating completed context {context}; no GPU rerun", flush=True)
        raw, guard = [json.loads(prefix.with_suffix(suffix).read_text()) for suffix in (".json", ".guard.json")]
        assert guard["command"] == command and raw["genai_native"]["sha256"] == qualified["runtime_sha256"]
        assert not raw["profiling"] and raw["comparison"] is None
        assert raw["load_experiment_flags"][GPU_FLAG] == "1" and raw["load_experiment_flags"][CPU_FLAG] == "1"
        assert raw["load_experiment_flags"]["ORT_GENAI_CPU_GREEDY_ARGMAX_AVX512"] == "0"
        verify_workers(raw, "control")
        first_three = reference_subset(raw, context)
        references = [sample for sample in controls["samples"] if sample["context"] == context]
        verify_result(first_three, guard, dict(provider_sha256=qualified["provider_sha256"]),
                      parent, references, context, True)
        for sample in raw["samples"]:
            assert sample["output_tokens"] == 256
            if not sample["warmup"]:
                assert all(phase["major_page_faults"] == 0 for phase in sample["process_resource_usage"].values())
        row = metrics(raw, guard, context, REQUESTS)
        row["frozen_reference_requests_checked"] = 3
        row["additional_requests_without_frozen_output_references"] = REQUESTS - 3
        rows.append(row)
        evidence.extend(prefix.with_suffix(suffix) for suffix in (".json", ".guard.json", ".log"))
        report = dict(status="completed" if len(rows) == 3 else "in_progress", rows=rows,
            configuration="qualified_dense_columns8_double_buffer_cpu_local8_full_validation",
            provider_sha256=qualified["provider_sha256"], runtime_sha256=qualified["runtime_sha256"],
            measured_requests_per_context=REQUESTS, excluded_warmups_per_context=1,
            ttft_percentile_method="linear_interpolation", prefill_definition="input_tokens/append_tokens_wall_time",
            decode_definition="sum(output_tokens-1)/sum(decode_seconds)",
            peak_gpu_definition="NVML_device_used_sampled_every_0.5s_including_load_and_warmup",
            quality_qualified=False, source_and_evidence_sha256={str(path): sha(path) for path in evidence})
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(table(rows), flush=True)

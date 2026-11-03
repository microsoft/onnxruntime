import argparse
import json
from pathlib import Path
import subprocess

from model_cache_preflight import prepare_model_cache, snapshot_model_cache
from qualify_quickgelu_mul_long_context import ROOT, sha
from qsa_rank_decode_windows import CPU_LOCAL_FLAG
from run_prefill256_model import verify_metadata
from run_qsa_score16_model import verify_paired


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, choices=(2048, 8192, 32768), default=2048)
    parser.add_argument("--full-validation", action="store_true")
    args = parser.parse_args()
    prefix = ROOT / f"cpu-local-v1-chunk512-gpu7-{'full-' if args.full_validation else ''}windows{args.context}-20261030"
    if any(prefix.with_suffix(s).exists() for s in (".json", ".guard.json", ".qualification.json")):
        raise FileExistsError(prefix)
    parent_path = ROOT / "prefill512-v3-contexts-20261030.qualification.json"
    parent = json.loads(parent_path.read_text())
    assert parent["all_original_sequences_match"] and parent["zero_major_faults"]
    provider = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill512/build-v3/libonnxruntime_providers_webgpu.so")
    assert sha(provider) == parent["provider_sha256"]
    topology = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True)
    gpu7 = next(line for line in topology.splitlines() if line.startswith("GPU7"))
    assert gpu7.split()[-3:-1] == ["48-95", "1"], gpu7
    topology_path = prefix.with_suffix(".topology.txt")
    with topology_path.open("x") as output:
        output.write(topology)
    if not args.full_validation:
        proof = json.loads((ROOT / f"cpu-local-v1-chunk512-gpu7-full-windows{args.context}-20261030.qualification.json").read_text())
        assert proof["zero_major_faults"] and proof["provider_sha256"] == parent["provider_sha256"]
    baseline = "prefill512-v3-gpu7-full-standard2048-20261030"
    command = json.loads((ROOT / f"{baseline}.guard.json").read_text())["command"]
    marker = "--tokens 256 --repetitions 3"
    assert command[-1].count(marker) == command[-1].count("--contexts 2048") == 1
    command[-1] = (command[-1].replace(baseline, prefix.name)
                   .replace("--contexts 2048", f"--contexts {args.context}")
                   .replace(marker, "--tokens 288 --repetitions 4 --compare-cpu-local-windows"))
    if not args.full_validation:
        assert command[-1].count(" --webgpu-validation-mode full") == 1
        command[-1] = command[-1].replace(" --webgpu-validation-mode full", "")
    model = Path("/home/sayanshaw/repos/qwen38-int4-mlp-input-projections/model")
    cache = prepare_model_cache(model, prefix)
    subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
                    "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "2400",
                    "--output-prefix", str(prefix), "--", *command], check=True)
    cache["after_run"] = snapshot_model_cache(model)
    Path(cache["evidence_path"]).write_text(json.dumps(cache, indent=2) + "\n")
    raw = json.loads(prefix.with_suffix(".json").read_text())
    guard = json.loads(prefix.with_suffix(".guard.json").read_text())
    verify_metadata(raw, guard, parent, parent, args.full_validation, chunk=512)
    assert guard["command"] == command and not raw["profiling"]
    affinity = raw["window_protocol"]["cpu_affinity"]
    assert affinity["candidate_cpus"] == list(range(48, 96))
    assert affinity["control_cpus"] == list(range(96))
    assert raw["window_protocol"]["cpu_affinity_restored"]
    for flag in ("ORT_WEBGPU_QMOE_PREFILL512", "ORT_WEBGPU_DENSE_TILE4",
                 "ORT_WEBGPU_HC_UP_PARALLEL2", "ORT_WEBGPU_QSA_SCORE16"):
        assert raw["load_experiment_flags"][flag] == "1"
    controls = json.loads((ROOT / "int4-pending64-standard-contexts.json").read_text())
    references = [s for s in controls["samples"] if s["context"] == args.context]
    result = verify_paired(raw, references, args.context, prefix="cpu_local", experiment_flag=CPU_LOCAL_FLAG)
    report = dict(status="cpu_local_original_sequences_qualified", context=args.context,
                  full_validation=args.full_validation, chunk=512, zero_major_faults=True,
                  all_original_prefixes_match=True, all_repeated_288_tokens_match=True,
                  provider_sha256=parent["provider_sha256"], graph_sha256=parent["graph_sha256"],
                  config_sha256=parent["config_sha256"], result=result, affinity=affinity,
                  cache_preflight=cache, model_provider_and_weights_unchanged=True,
                  quality_qualified=False, headline_eligible=False,
                  source_sha256={p.name: sha(p) for p in (parent_path, topology_path,
                      prefix.with_suffix(".json"), prefix.with_suffix(".guard.json"),
                      Path(__file__), ROOT / "cpu_local_windows.py", ROOT / "qsa_rank_decode_windows.py",
                      ROOT / "benchmark_webgpu_int4.py")})
    prefix.with_suffix(".qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "context", "full_validation", "result")}), flush=True)


if __name__ == "__main__":
    main()

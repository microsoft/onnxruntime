import json
import math
from pathlib import Path
import subprocess

from qualify_quickgelu_mul_long_context import (
    ROOT, identities, load, sha, snapshot,
)

GPU_UUID = "GPU-0e70e969-bbfc-80c5-d582-88be134632ea"
PREFIX = ROOT / "quickgelu-mul-gpu7-8k-20261028"
OUTPUT = PREFIX.with_suffix(".qualification.json")


def validate_gpu7(path, context, control, identity):
    data, guard = load(path), load(path.with_suffix(".guard.json"))
    assert data["status"] == guard["status"] == "completed" and guard["exit_code"] == 0
    assert guard["gpu_index"] == 7 and guard["gpu_uuid"] == GPU_UUID
    assert data["provider"] == "webgpu" and data["chunk"] == 16 and not data["profiling"]
    assert data["webgpu_max_pending_dispatches_override"] == 64
    assert not data["quality_qualified"] and not data["headline_eligible"]
    for key in ("plugin_sha256", "graph_sha256", "config_sha256"):
        assert data[key] == identity["provider_sha256" if key == "plugin_sha256" else key]
    log = path.with_suffix(".log").read_text()
    assert "GetCapability()" not in log and "FAIL" not in log
    invocation = " ".join(guard["command"])
    assert "ORT_WEBGPU_HC_EXACT_DENSE=1" in invocation and "ORT_WEBGPU_EXACT_DENSE_MID=1" in invocation
    samples = data["samples"]
    references = [s for s in control["samples"] if s["context"] == context]
    assert len(samples) == len(references) == 4
    assert [s["warmup"] for s in samples] == [True, False, False, False]
    for sample, reference in zip(samples, references):
        for key in ("context", "repetition", "warmup", "seed", "offset", "input_tokens",
                    "prompt_sha256", "output_token_ids", "output_sha256"):
            assert sample[key] == reference[key], key
        assert sample["input_tokens"] == context
        assert sample["output_tokens"] == len(sample["output_token_ids"]) == 256
        assert sample["decode_seconds"] > 0
        assert math.isclose(sample["decode_tps"], 255 / sample["decode_seconds"], rel_tol=1e-12)
        assert all(v["major_page_faults"] == 0 for v in sample["process_resource_usage"].values())
    summary = data["summary"][f"{context}:configured"]
    return dict(context_tokens=context, median_decode_tps=summary["median_decode_tps"],
                aggregate_decode_tps=summary["decode_tps"],
                sample_decode_tps=[s["decode_tps"] for s in samples[1:]],
                original_sequences_exact_including_warmup=4,
                all_major_page_faults_zero=True, benchmark=str(path),
                guard=str(path.with_suffix(".guard.json")), matched_control=False)


def main():
    if OUTPUT.exists() or any(
        PREFIX.with_suffix(suffix).exists() for suffix in (".json", ".guard.json", ".log")
    ):
        raise FileExistsError(PREFIX)
    control = load(ROOT / "int4-pending64-standard-contexts.json")
    identity = identities(control)
    invocation = load(ROOT / "quickgelu-mul-candidate-32k-20261027.guard.json")["command"].copy()
    invocation[-1] = invocation[-1].replace(
        "GPU-c1593c4e-0f5f-7bf7-3567-6c3c19703afd", GPU_UUID
    ).replace("quickgelu-mul-candidate-32k-20261027", PREFIX.name).replace(
        "--contexts 32768", "--contexts 8192"
    )
    report = dict(status="running", initial_resources=snapshot(), identity=identity,
                  gpu_index=7, gpu_uuid=GPU_UUID, quality_qualified=False,
                  headline_eligible=False, driver_sha256=sha(Path(__file__)))
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    try:
        subprocess.run(
            ["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
             "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "1000",
             "--output-prefix", str(PREFIX), "--", *invocation], check=True,
        )
        report["row"] = validate_gpu7(PREFIX.with_suffix(".json"), 8192, control, identity)
        assert identities(control) == identity
        report["status"] = "original_8k_sequences_qualified_gpu7"
    except BaseException as error:
        report.update(status="failed_or_interrupted", error=repr(error))
        raise
    finally:
        report["final_resources"] = snapshot()
        report["own_resources_released"] = (
            GPU_UUID not in report["final_resources"]["compute_owners"]
            and not report["final_resources"]["private_containers"]
        )
        OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    assert report["own_resources_released"]
    print(json.dumps(report["row"], indent=2))


if __name__ == "__main__":
    main()

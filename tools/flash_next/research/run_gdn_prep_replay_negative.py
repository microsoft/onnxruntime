import json
from pathlib import Path
import shlex
import subprocess

from build_gdn_prep_node_replay import FLAG
from qualify_quickgelu_mul_long_context import ROOT, sha
from run_gdn_prep_replay_native import DIRECTORY


def main():
    prefix = ROOT / "gdn-prep-node-replay-v2-negative-gpu7-20261101"
    assert not prefix.with_suffix(".guard.json").exists()
    comparator = Path("/home/sayanshaw/repos/onnxruntime-webgpu-h200-perf/build-int4-gdn-prep/gdn_prep_compare")
    provider = DIRECTORY / "libonnxruntime_providers_webgpu.so"
    commands = ["set -e", "/workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe "
                "GPU-0e70e969-bbfc-80c5-d582-88be134632ea"]
    evidence = []
    cases = [(mode, name, "GdnProjPrepF16 requires ab [B,L,96]")
             for mode in (0, 1) for name in ("bad_state_width", "bad_state_channels")]
    cases.append((2, "b1_l1", f"{FLAG} must be 0 or 1"))
    for mode, name, error in cases:
        log = ROOT / f"{prefix.name}-mode{mode}-{name}.log"
        assert not log.exists()
        fixture = ROOT / "gdn-prep-fixtures" / f"gdn_prep_{name}.onnx"
        argv = [str(comparator).replace("/home/sayanshaw/repos", "/workspace", 1),
                str(provider).replace("/home/sayanshaw/repos", "/workspace", 1),
                "/evidence/webgpu-perf-oct20/gdn-prep-fixtures/" + fixture.name]
        target = "/evidence/webgpu-perf-oct20/" + log.name
        commands.append(f"if env {FLAG}={mode} ORT_WEBGPU_STATIC_NODE_REPLAY=1 "
                        f"{shlex.join(argv)} > {shlex.quote(target)} 2>&1; then "
                        "echo 'FAIL: invalid input or selector accepted'; exit 1; fi")
        commands.append(f"grep -F {shlex.quote(error)} {shlex.quote(target)}")
        evidence.extend((log, fixture))
    command = "; ".join(commands)
    subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
        "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "180",
        "--output-prefix", str(prefix), "--", "bash", "-lc", command], check=True)
    guard = json.loads(prefix.with_suffix(".guard.json").read_text())
    assert guard["status"] == "completed" and guard["exit_code"] == 0
    report = dict(status="gdn_prep_replay_original_layout_and_fresh_selector_rejections_qualified",
        provider_sha256=sha(provider), modes_tested=[0, 1], invalid_selector_fresh_process=True,
        invalid_state_width_and_channels=True,
        source_sha256={str(p): sha(p) for p in (
            comparator, *evidence, prefix.with_suffix(".guard.json"), prefix.with_suffix(".log"))})
    prefix.with_suffix(".qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "invalid_selector_fresh_process")}))


if __name__ == "__main__":
    main()

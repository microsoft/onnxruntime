import json
from pathlib import Path
import subprocess
import time

from build_qmoe_node_replay import sha
from prepare_webgpu_int4_metrics import ROOT


def execute(script, *arguments):
    command = ["python3", "-u", str(ROOT / script), *arguments]
    print(json.dumps(dict(stage="starting", command=command)), flush=True)
    subprocess.run(command, check=True)


def progression(script, variant, stamp, initial_status, repeat_status):
    for context in (2048, 8192, 32768):
        for repeat in (False, True):
            execute(script, "--context", str(context), *(["--repeat"] if repeat else []))
            path = ROOT / f"{variant}-context{context}-{'repeat-' if repeat else ''}screen-{stamp}.qualification.json"
            report = json.loads(path.read_text())
            for name, digest in report["source_sha256"].items():
                assert sha(Path(name)) == digest, name
            print(json.dumps(dict(stage="screen_result", variant=variant,
                                  context=context, repeat=repeat, status=report["status"],
                                  rows=report["rows"])), flush=True)
            expected = repeat_status if repeat else initial_status
            if report["status"] != expected:
                assert report["status"] == "no_supported_decode_gain"
                print(json.dumps(dict(stage="held_default_off", variant=variant,
                                      reason="strict matched consistency gate failed",
                                      unchanged_retry=False)), flush=True)
                return False
    return True


if __name__ == "__main__":
    outcome = ROOT / "hc-parallel16-v1-native-attempt3-outcome.json"
    deadline = time.monotonic() + 2700
    while not outcome.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Native prerequisite did not finalize: {outcome}")
        time.sleep(5)
    native = json.loads(outcome.read_text())
    assert native["status"] == "hc_separate_original_outputs_supported", native["status"]
    for name, digest in native["source_and_evidence_sha256"].items():
        assert sha(Path(name)) == digest, name
    execute("run_hc_parallel16_invalid.py")
    execute("run_hc_parallel16_model.py", "--mode", "candidate", "--context", "2048",
            "--phase", "profile", "--full-validation")
    cpu_supported = progression(
        "run_cpu_greedy_argmax_avx512_screen.py", "cpu-greedy-argmax-avx512-local8-v1", "20261103",
        "initial_cpu_greedy_decode_gain_both_modes", "repeated_cpu_greedy_decode_gain_both_modes")
    hc_supported = progression(
        "run_hc_parallel16_cpu_local8_screen.py", "hc-parallel16-cpu-local8-v1", "20261103",
        "initial_hc_parallel16_cpu_local8_decode_gain_both_modes",
        "repeated_hc_parallel16_cpu_local8_decode_gain_both_modes")
    print(json.dumps(dict(stage="progression_completed", cpu_supported_all_contexts=cpu_supported,
                          hc_supported_all_contexts=hc_supported, candidates_stacked=False,
                          quality_qualified=False, automatic_default_promotion=False)), flush=True)

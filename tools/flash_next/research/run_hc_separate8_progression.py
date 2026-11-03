import json
from pathlib import Path
import time

from build_qmoe_node_replay import sha
from prepare_webgpu_int4_metrics import ROOT
from run_avx512_hc_parallel16_progression import execute, progression


if __name__ == "__main__":
    outcome = ROOT / "hc-separate8-v1-native-outcome.json"
    deadline = time.monotonic() + 2700
    while not outcome.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Native prerequisite did not finalize: {outcome}")
        time.sleep(5)
    native = json.loads(outcome.read_text())
    assert native["status"] == "hc_separate_original_outputs_supported", native["status"]
    for name, digest in native["source_and_evidence_sha256"].items():
        assert sha(Path(name)) == digest, name
    execute("run_hc_separate8_invalid.py")
    execute("run_hc_separate8_model.py", "--mode", "candidate", "--context", "2048",
            "--phase", "profile", "--full-validation")
    supported = progression(
        "run_hc_separate8_cpu_local8_screen.py", "hc-separate8-cpu-local8-v1", "20261103",
        "initial_hc_separate8_cpu_local8_decode_gain_both_modes",
        "repeated_hc_separate8_cpu_local8_decode_gain_both_modes")
    print(json.dumps(dict(stage="progression_completed", supported_all_contexts=supported,
                          candidates_stacked=False, quality_qualified=False,
                          automatic_default_promotion=False)), flush=True)

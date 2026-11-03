import json
from pathlib import Path
import subprocess
import time

from build_qmoe_node_replay import sha
from prepare_webgpu_int4_metrics import ROOT
from run_gpu_slice_model import verify_sources


if __name__ == "__main__":
    state_path = ROOT / "bind-group-inline-key-v2-queued-model.json"
    if state_path.exists():
        raise FileExistsError(state_path)
    generation = ROOT / "bind-group-inline-key-v2-model-generation.json"
    native = ROOT / "bind-group-inline-key-v2-native-outcome.json"
    verify_sources(generation, "inline_key_model_controller_specialized")
    verify_sources(native, "bind_group_inline_key_native_finite_bitwise_qualified")
    upstream = ROOT / "context-flag-cache-v1-gpu1-pipeline.json"
    deadline = time.monotonic() + 7200
    state = dict(status="waiting_for_owned_gpu1_pipeline", started_ns=time.time_ns(),
                 upstream=str(upstream), own_source_sha256=sha(Path(__file__)))

    def persist():
        state_path.write_text(json.dumps(state, indent=2) + "\n")

    persist()
    try:
        while True:
            try:
                status = json.loads(upstream.read_text())["status"]
            except json.JSONDecodeError:
                print("Upstream controller is publishing state; retrying", flush=True)
                time.sleep(.1)
                continue
            if status in ("held_no_unchanged_retry", "bounded_2k_8k_complete_32k_requires_supported_control"):
                break
            if status == "failed":
                raise RuntimeError("Owned upstream pipeline failed; preserve evidence rather than silently continuing")
            if time.monotonic() >= deadline:
                raise TimeoutError("Owned upstream pipeline did not finish within the bounded wait")
            time.sleep(5)
        verify_sources(generation, "inline_key_model_controller_specialized")
        verify_sources(native, "bind_group_inline_key_native_finite_bitwise_qualified")
        state.update(status="running_inline_key_model", upstream_terminal=status)
        persist()
        subprocess.run(["python3", "-u", str(ROOT / "run_bind_group_inline_key_model_v2.py")], check=True)
        result = json.loads((ROOT / "bind-group-inline-key-v2-gpu1-pipeline.json").read_text())
        state.update(status="completed", model_terminal=result["status"], completed_ns=time.time_ns())
        persist()
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        persist()
        raise

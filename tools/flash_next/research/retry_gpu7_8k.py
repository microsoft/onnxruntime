import json
from pathlib import Path
import subprocess
import time

from qualify_quickgelu_mul_long_context import ROOT, identities, load, sha, snapshot
from resume_gpu7_8k import validate_gpu7


PREFIX = ROOT / "quickgelu-mul-gpu7-8k-cold-retry-20261028"
OUTPUT = PREFIX.with_suffix(".qualification.json")


def main():
    if OUTPUT.exists() or any(
        PREFIX.with_suffix(s).exists() for s in (".json", ".guard.json", ".log")
    ):
        raise FileExistsError(PREFIX)
    original = ROOT / "quickgelu-mul-gpu7-8k-20261028"
    deadline = time.monotonic() + 1100
    while True:
        previous = load(original.with_suffix(".guard.json"))
        if previous["status"] not in ("running", "waiting_for_idle_gpu"):
            break
        if time.monotonic() > deadline:
            raise TimeoutError("Original guarded run did not finish")
        time.sleep(5)
    control = load(ROOT / "int4-pending64-standard-contexts.json")
    identity = identities(control)
    if previous["status"] == "completed":
        row = validate_gpu7(original.with_suffix(".json"), 8192, control, identity)
        OUTPUT.write_text(json.dumps(dict(status="original_8k_sequences_qualified_gpu7",
                                         reused_successful_original=True, row=row,
                                         identity=identity, quality_qualified=False,
                                         headline_eligible=False), indent=2) + "\n")
        print(json.dumps(row, indent=2))
        return
    if previous["status"] != "run_timed_out":
        raise RuntimeError(f"Refusing automatic retry after non-timeout failure: {previous['status']}")
    invocation = previous["command"].copy()
    invocation[-1] = invocation[-1].replace(original.name, PREFIX.name)
    report = dict(status="running", initial_resources=snapshot(), identity=identity,
                  previous_attempt_guard=str(original.with_suffix(".guard.json")),
                  previous_attempt_status=previous["status"],
                  reason="Cold disk loading exceeded original 1000-second guard; fresh 1800-second bounded retry.",
                  driver_sha256=sha(Path(__file__)), quality_qualified=False, headline_eligible=False)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    try:
        subprocess.run(
            ["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
             "--idle-seconds", "3", "--wait-seconds", "30", "--run-seconds", "1800",
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
        OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["row"], indent=2))


if __name__ == "__main__":
    main()

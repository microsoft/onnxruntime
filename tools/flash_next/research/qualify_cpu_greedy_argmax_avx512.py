import json
import math
from pathlib import Path
import re
import subprocess

from build_qmoe_node_replay import sha


ROOT = Path(__file__).parent
BUILD = Path("/home/sayanshaw/repos/qwen38-int4-cpu-greedy-argmax-avx512/build-v1")


def host_path(name):
    return Path(name.replace("/workspace/", "/home/sayanshaw/repos/")
                    .replace("/evidence/", str(ROOT.parent) + "/"))


if __name__ == "__main__":
    output = ROOT / "cpu-greedy-argmax-avx512-v1-cpu.qualification.json"
    log_path = ROOT / "cpu-greedy-argmax-avx512-v1-cpu.log"
    if output.exists() or log_path.exists():
        raise FileExistsError("CPU proof or execution log already exists")
    manifest_path = BUILD / "build-manifest.json"
    build = json.loads(manifest_path.read_text())
    runtime = BUILD / "libonnxruntime-genai.so"
    executable = BUILD / "cpu_greedy_argmax_compare"
    assert build["baseline_relink_byte_identical"] and build["baseline_search_rebuild_byte_identical"]
    assert sha(runtime) == build["runtime_sha256"]
    sources = [Path(__file__), manifest_path, runtime]
    for name, digest in build["source_sha256"].items():
        path = host_path(name)
        assert sha(path) == digest, path
        sources.append(path)
    command = ["taskset", "-c", "48-55", str(executable)]
    with log_path.open("x") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, text=True)
    result.check_returncode()
    log = log_path.read_text()
    assert "FAIL" not in log and "NOT_MODEL_TPS" in log
    match = re.search(r"PASS cases=(\d+) elements=(\d+) selected_ids_exact=1 "
                      r"signed_zeros_nan_inf_tested=1 input_unchanged=1", log)
    assert match is not None
    assert (int(match[1]), int(match[2])) == (270337, 32271653)
    trials = [dict(trial=int(trial), lanes8_before_seconds=float(before), avx512_seconds=float(candidate),
                   lanes8_after_seconds=float(after), matched_speedup=float(speedup))
              for trial, before, candidate, after, speedup in re.findall(
                  r"CPU_ONLY trial=(\d+) lanes8_before_s=([0-9.e+-]+) avx512_s=([0-9.e+-]+) "
                  r"lanes8_after_s=([0-9.e+-]+) matched_speedup=([0-9.e+-]+) vocab=248320 calls=200", log)]
    assert [trial["trial"] for trial in trials] == list(range(5))
    assert all(math.isfinite(value) and value > 0
               for trial in trials for key, value in trial.items() if key != "trial")
    sources.append(log_path)
    proof = dict(status="cpu_greedy_avx512_selected_ids_exact", command=command,
        runtime_sha256=build["runtime_sha256"], parent_runtime_sha256=build["parent_runtime_sha256"],
        original_runtime_sha256=build["original_runtime_sha256"], cases=int(match[1]), elements=int(match[2]),
        selected_ids_match_original_and_lanes8=True, inputs_unchanged=True, cpu_affinity=list(range(48, 56)),
        trials=trials, model_qualified=False, quality_qualified=False, model_tps_gain_claimed=False,
        default_enabled=False, source_sha256={str(path): sha(path) for path in sources})
    with output.open("x") as stream:
        stream.write(json.dumps(proof, indent=2) + "\n")
    print(log, end="")
    print(json.dumps({key: proof[key] for key in ("status", "cases", "elements", "runtime_sha256")}))

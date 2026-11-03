import json
from pathlib import Path
import re

from build_qmoe_node_replay import sha


ROOT = Path(__file__).parent
BUILD = Path("/home/sayanshaw/repos/qwen38-int4-cpu-greedy-argmax/build-v1")


if __name__ == "__main__":
    output = ROOT / "cpu-greedy-argmax-lanes8-v1-cpu.qualification.json"
    if output.exists():
        raise FileExistsError(output)
    build_path = BUILD / "build-manifest.json"
    build = json.loads(build_path.read_text())
    log_path = ROOT / "cpu-greedy-argmax-lanes8-v1-cpu.log"
    log = log_path.read_text()
    assert "FAIL" not in log
    assert "PASS cases=127233 elements=21358629 selected_ids_exact=1 " in log
    rows = re.findall(r"CPU_ONLY trial=(\d+) original_before_s=([\d.]+) lanes8_s=([\d.]+) "
                      r"original_after_s=([\d.]+) matched_speedup=([\d.]+) vocab=248320 calls=200", log)
    assert len(rows) == 5 and [int(row[0]) for row in rows] == list(range(5))
    assert all(float(value) > 0 for row in rows for value in row[1:])
    assert all(float(row[-1]) > 1 for row in rows)
    runtime = BUILD / "libonnxruntime-genai.so"
    assert sha(runtime) == build["runtime_sha256"]
    source_hashes = {}
    for name, digest in build["source_sha256"].items():
        path = Path(name.replace("/workspace", "/home/sayanshaw/repos", 1)
                    .replace("/evidence", str(ROOT.parent), 1))
        assert sha(path) == digest, name
        source_hashes[str(path)] = digest
    report = dict(status="cpu_greedy_lanes8_selected_ids_exact", cases=127233,
        input_elements=21358629, selected_ids_exact=True, input_unchanged=True,
        signed_zeros_nan_inf_tested=True, cpu_only=True, not_model_tps=True,
        runtime_sha256=build["runtime_sha256"], cpu_timing_rows=rows,
        source_sha256={**source_hashes, **{str(path): sha(path) for path in
            (Path(__file__), build_path, log_path, runtime, ROOT / "test_cpu_greedy_argmax_lanes8.py")}})
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "cases", "input_elements")}), flush=True)

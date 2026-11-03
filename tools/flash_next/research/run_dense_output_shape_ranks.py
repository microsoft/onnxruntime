import json
from pathlib import Path
import shlex
import subprocess

from build_qmoe_node_replay import sha
from prepare_dense_output_shape_ranks import OUTPUT
from qualify_quickgelu_mul_long_context import ROOT
from run_dense_output_shape_native import PREFIX as NATIVE_PREFIX, SHAPE_STATS, verify_stats


PREFIX = ROOT / "dense-output-shape-v2-ranks-native-gpu7-20261031"


def main():
    for suffix in (".qualification.json", ".guard.json", ".log"):
        if PREFIX.with_suffix(suffix).exists():
            raise FileExistsError(PREFIX.with_suffix(suffix))
    native_path = NATIVE_PREFIX.with_suffix(".qualification.json")
    native = json.loads(native_path.read_text())
    assert native["status"] == "dense_output_shape_changing_inputs_finite_bitwise_qualified"
    verify_stats(native["stats"])
    manifest_path = OUTPUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "synthetic_rank_fixtures_prepared"
    for name, digest in manifest["source_sha256"].items():
        assert sha(Path(name)) == digest, name
    fixtures = manifest["fixtures"]
    assert len(fixtures) == 36 and {r["rank"] for r in fixtures} == {1, 2, 3, 4}
    for row in fixtures:
        assert sha(Path(row["path"])) == row["sha256"], row["path"]
    directory = Path("/home/sayanshaw/repos/qwen38-int4-dense-output-shape/build-v2")
    provider = directory / "libonnxruntime_providers_webgpu.so"
    assert sha(provider) == native["provider_sha256"]
    assert sha(directory / "native_compare") == native["comparator_sha256"]
    build_path = directory / "build-manifest.json"
    assert sha(build_path) == native["build_manifest_sha256"]
    build = json.loads(build_path.read_text())
    for name, digest in build["source_sha256"].items():
        assert sha(directory / name) == digest, name
    original_guard = json.loads(NATIVE_PREFIX.with_suffix(".guard.json").read_text())
    original_command = original_guard["command"]
    assert original_command[:2] == ["bash", "-lc"] and original_command[-1].count("exec ") == 1
    script = original_command[-1].split("exec ", 1)[0].replace(NATIVE_PREFIX.name, PREFIX.name)
    args = [directory / "native_compare", provider, *[Path(row["path"]) for row in fixtures]]
    script += "exec " + shlex.join(str(p).replace("/home/sayanshaw/repos", "/workspace") for p in args)
    command = ["bash", "-lc", script]
    subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
                    "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "900",
                    "--output-prefix", str(PREFIX), "--", *command], check=True)
    guard = json.loads(PREFIX.with_suffix(".guard.json").read_text())
    log = PREFIX.with_suffix(".log").read_text()
    assert guard["status"] == "completed" and guard["exit_code"] == 0 and guard["command"] == command
    assert "FAIL" not in log and "GetCapability" not in log
    assert log.count(" changing_rounds=128 invalid_flag_rejected=1") == 36
    comparisons = sum(row["N"] for row in fixtures) * 128
    assert comparisons == manifest["finite_bitwise_comparisons"] == 24543744
    assert f"PASS total_exact_elements={comparisons}; CPU fallback disabled" in log
    stats_path = ROOT / (PREFIX.name + "-stats.jsonl")
    rows = [json.loads(s) for s in stats_path.read_text().splitlines()]
    assert len(rows) == 36
    assert all(r["operator"] == "MatMulNBitsOutputShapeExact" and r["requests"] == 256 and
               r["hits"] == 255 and r["captures"] == 1 and r["entries"] == 1 and
               r["rebinds"] + r["binding_hits"] == r["hits"] for r in rows), rows
    shape_path = ROOT / (PREFIX.name + "-shape-stats.jsonl")
    shapes = [json.loads(s) for s in shape_path.read_text().splitlines()]
    assert len(shapes) == 36 and sum(r["fast_shape_hits"] == 128 for r in shapes) == 27
    assert sum(r["fast_shape_hits"] == 0 for r in shapes) == 9
    sources = [Path(__file__), native_path, SHAPE_STATS, manifest_path, build_path, stats_path, shape_path,
               PREFIX.with_suffix(".guard.json"), PREFIX.with_suffix(".log")]
    report = dict(status="dense_output_shape_rank1_fallback_rank234_finite_bitwise_qualified",
                  provider_sha256=native["provider_sha256"], finite_bitwise_comparisons=comparisons,
                  ranks=[1, 2, 3, 4], fast_shape_fixtures=27, rank1_fallback_fixtures=9,
                  pending_windows_tested=[1, 64], invalid_flag_rejections=36,
                  fixture_sha256={r["path"]: r["sha256"] for r in fixtures},
                  quality_qualified=False, headline_eligible=False,
                  source_sha256={str(p): sha(p) for p in sources})
    with PREFIX.with_suffix(".qualification.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "finite_bitwise_comparisons", "ranks")}), flush=True)


if __name__ == "__main__":
    main()

import collections
import json
from pathlib import Path
import shlex
import subprocess

from build_dsa_four_subgroups import FLAG, NAME
from build_qmoe_node_replay import sha
from run_dense_columns16_synthetic import ORIGINAL, ROOT, UUID, run_guard, workspace


BUILD = Path("/home/sayanshaw/repos/qwen38-int4-dsa-four-subgroups/build-v1")
PROOF = BUILD.parent / "native-proof-v1"
DATA = BUILD.parent / "original-references-v1"
FLAGS = ("ORT_WEBGPU_DSA_QUERY_CACHE=1 ORT_WEBGPU_DYNAMIC_SPARSE_ATTENTION_SPLIT_KV=1 "
         "ORT_WEBGPU_DYNAMIC_UNIFORMS=1 ORT_WEBGPU_UNIFORM_ARENA=1 "
         "ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES=8192")


if __name__ == "__main__":
    outcome = ROOT / "dsa-four-subgroups-v1-native-outcome.json"
    if outcome.exists() or DATA.exists():
        raise FileExistsError("Existing native evidence")
    build_path, comparator_path = BUILD / "build-manifest.json", PROOF / "build-manifest.json"
    fixture_path = ROOT / "dsa-query-cache-v2-native-gpu7-20261030-fixtures.json"
    build, comparator, fixtures = [json.loads(path.read_text())
                                  for path in (build_path, comparator_path, fixture_path)]
    assert build["baseline_byte_identical"] and build["all_152_original_source_objects_byte_identical"]
    provider, binary = BUILD / "libonnxruntime_providers_webgpu.so", PROOF / "independent_compare"
    assert sha(provider) == build["provider_sha256"] and sha(ORIGINAL) == build["baseline_provider_sha256"]
    assert sha(binary) == comparator["comparator_sha256"]
    evidence = [Path(__file__), build_path, comparator_path, fixture_path, provider, binary, ORIGINAL]
    for name, digest in build["source_sha256"].items():
        path = BUILD / name
        assert sha(path) == digest, path
        evidence.append(path)
    for name, digest in comparator["copied_sha256"].items():
        assert sha(PROOF / name) == digest
        evidence.append(PROOF / name)
    directory = ROOT / "dsa-split-kv-fixtures"
    for name, digest in fixtures["fixture_sha256"].items():
        assert sha(directory / name) == digest, name
        evidence.append(directory / name)
    assert len(fixtures["output_elements"]) == 27
    DATA.mkdir()
    captured, records, coverage = {}, [], []
    status = "dsa_four_subgroups_independent_original_outputs_supported"
    for pending in (1, 64):
        for phase, library, mode, selected in (("capture", ORIGINAL, "capture", 0),
                                              ("default", provider, "compare", 0),
                                              ("candidate", provider, "compare", 1)):
            for name, digest in captured.items():
                assert sha(Path(name)) == digest, name
            prefix = ROOT / f"dsa-four-subgroups-v1-pending{pending}-{phase}-gpu7"
            commands = [f"set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe {UUID}",
                        f"export {FLAGS} {FLAG}={selected}"]
            for name in fixtures["output_elements"]:
                reference = DATA / f"pending{pending}-{Path(name).stem}.reference.bin"
                profile = "/evidence/webgpu-perf-oct20/" + prefix.name + "-" + Path(name).stem
                commands.append(shlex.join([workspace(binary), mode, workspace(library),
                    "/evidence/webgpu-perf-oct20/dsa-split-kv-fixtures/" + name,
                    workspace(reference), profile, str(pending)]))
            subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.used,utilization.gpu",
                            "--format=csv,noheader"], check=True)
            subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                            "--format=csv,noheader"], check=True)
            code, log = run_guard("; ".join(commands), prefix)
            records.append(dict(pending=pending, phase=phase, exit_code=code))
            evidence.extend([prefix.with_suffix(".log"), prefix.with_suffix(".guard.json")])
            if code == 2:
                status = f"held_dsa_four_subgroups_pending{pending}_{phase}_bitwise_mismatch"
                break
            assert code == 0 and log.count("mismatches=0 repeats=9") == 27
            profiles = [ROOT / Path(line.removeprefix("PROFILE ")).name
                        for line in log.splitlines() if line.startswith("PROFILE ")]
            assert len(profiles) == 27
            counts = collections.Counter()
            for profile in profiles:
                events = json.loads(profile.read_text())
                selected_count = sum(event.get("cat") == "Kernel" and
                                     event["name"].endswith("&" + NAME) for event in events)
                if phase != "candidate":
                    assert selected_count == 0
                if "-model_2k_" in profile.name:
                    assert selected_count == (9 if phase == "candidate" else 0)
                counts.update(event["name"].rsplit("&", 1)[-1]
                              for event in events if event.get("cat") == "Kernel")
                evidence.append(profile)
            if phase == "candidate":
                assert counts[NAME] > 0 and counts["DynamicSparseAttention"] > 0
            coverage.append(dict(pending=pending, phase=phase, program_counts=dict(counts)))
            if phase == "capture":
                for name in fixtures["output_elements"]:
                    reference = DATA / f"pending{pending}-{Path(name).stem}.reference.bin"
                    captured[str(reference)] = sha(reference)
                    evidence.append(reference)
            print(json.dumps(records[-1]), flush=True)
        if status.startswith("held_"):
            break
    report = dict(status=status, records=records, coverage=coverage, fixtures=27,
        comparisons_per_complete_phase=9 * sum(fixtures["output_elements"].values()),
        captured_sha256=captured, changing_inputs_tested=False, fixed_fixture_repeats=9,
        full_model_coverage=False, quality_qualified=False, default_enabled=False,
        source_and_evidence_sha256={str(path): sha(path) for path in evidence})
    outcome.write_text(json.dumps(report, indent=2) + "\n")
    print(status, flush=True)

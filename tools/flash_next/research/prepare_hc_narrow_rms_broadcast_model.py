import ast
from pathlib import Path

from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent


def adapt(source):
    for before, after in (
        ("qwen38-int4-dense-columns8-double-buffer/build-v1",
         "qwen38-int4-hc-narrow-rms-broadcast/build-v3"),
        ("dense-columns8-double-buffer-v1", "hc-narrow-rms-broadcast-v3"),
        ("dense-columns8-double-buffer-cpu-local8-v1", "hc-narrow-rms-broadcast-cpu-local8-v3"),
        ("run_dense_columns8_double_buffer_cpu_local8_standard",
         "run_hc_narrow_rms_broadcast_cpu_local8_standard"),
        ("build_dense_columns8_double_buffer import", "build_hc_narrow_rms_broadcast_v3 import"),
        ("columns8_double_buffer_cpu_local8_", "hc_narrow_rms_broadcast_cpu_local8_"),
        ("-columns8-gpu7", "-candidate-gpu7"),
        ('"columns8"', '"candidate"'), ("'columns8'", "'candidate'"),
    ):
        source = source.replace(before, after)
    return source


def model_source():
    source = adapt((ROOT / "run_dense_columns8_double_buffer_model.py").read_text())
    start = source.index("def verify_profile(")
    stop = source.index("def main():", start)
    source = source[:start] + '''def verify_profile(counts, mode):
    selected = "HCPostNormProjectionNarrowRmsBroadcastF16" if mode == "candidate" else "HCPostNormProjectionF16"
    original = "HCPostNormProjectionF16" if mode == "candidate" else "HCPostNormProjectionNarrowRmsBroadcastF16"
    assert counts[selected] == 94 and counts[original] == 0, counts
    assert counts["MatMulNBitsBlock32F16"] == 159, counts
    assert counts["DenseColumns8DoubleBuffer"] == 0, counts
    assert counts["DynamicSparseAttentionQueryCache"] == 12, counts
    assert counts["DynamicSparseAttention"] == 0, counts


''' + source[stop:]
    start = source.index("    native_path = ROOT /")
    stop = source.index("    parent = json.loads(parent_path.read_text())", start)
    source = source[:start] + '''    native_path = ROOT / "hc-narrow-rms-broadcast-v3-native-outcome.json"
    invalid_path = ROOT / "hc-narrow-rms-broadcast-v3-invalid-outcome.json"
    native_proof = json.loads(native_path.read_text())
    invalid = json.loads(invalid_path.read_text())
    assert native_proof["status"] == "hc_separate_original_outputs_supported"
    assert [row["phase"] for row in native_proof["records"]] == ["capture", "default", "candidate"]
    assert all(row["exit_code"] == 0 for row in native_proof["records"])
    assert native_proof["independent_original_provider_captures"]
    assert native_proof["fixtures"] == 36 and native_proof["rounds_per_fixture"] == 128
    assert native_proof["comparisons_per_complete_phase"] == 607143936
    assert len(native_proof["captured_sha256"]) == 36
    assert len(native_proof["replay_activity"]) == 3
    assert all(row["instances"] == 12 and row["binding_hits"] > 0
               for row in native_proof["replay_activity"])
    assert invalid["status"] == "fresh_invalid_hc_narrow_rms_broadcast_selector_rejected"
    for proof in (native_proof, invalid):
        for path, digest in proof["source_and_evidence_sha256"].items():
            assert sha(Path(path)) == digest, path
''' + source[stop:]
    source = source.replace("native_path, synthetic_path, invalid_path", "native_path, invalid_path")
    return source.replace('ROOT / "prepare_dense_columns8_double_buffer_model.py"',
                          'ROOT / "prepare_hc_narrow_rms_broadcast_model.py"')


def standard_source():
    source = adapt((ROOT / "run_dense_columns8_double_buffer_cpu_local8_standard.py").read_text())
    source = replace_once(source,
        'assert gpu["result"]["program_counts"]["DenseColumns8DoubleBuffer"] == 158',
        'assert gpu["result"]["program_counts"]["HCPostNormProjectionNarrowRmsBroadcastF16"] == 94\n'
        '    assert gpu["result"]["program_counts"].get("HCPostNormProjectionF16", 0) == 0')
    return replace_once(source,
        'assert gpu["result"]["program_counts"]["MatMulNBitsBlock32F16"] == 1',
        'assert gpu["result"]["program_counts"]["MatMulNBitsBlock32F16"] == 159')


def screen_source():
    return adapt((ROOT / "run_dense_columns8_double_buffer_cpu_local8_screen.py").read_text())


if __name__ == "__main__":
    for name, source in (
        ("run_hc_narrow_rms_broadcast_model.py", model_source()),
        ("run_hc_narrow_rms_broadcast_cpu_local8_standard.py", standard_source()),
        ("run_hc_narrow_rms_broadcast_cpu_local8_screen.py", screen_source()),
    ):
        ast.parse(source)
        with (ROOT / name).open("x") as stream:
            stream.write(source)
        print(name)

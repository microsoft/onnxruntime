import ast
from pathlib import Path

from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent


def adapt(source):
    for before, after in (
        ("qwen38-int4-hc-narrow-rms-broadcast/build-v3", "qwen38-int4-qmoe-fc1-double-buffer/build-v1"),
        ("hc-narrow-rms-broadcast-v3", "qmoe-fc1-double-buffer-v1"),
        ("hc-narrow-rms-broadcast-cpu-local8-v3", "qmoe-fc1-double-buffer-cpu-local8-v1"),
        ("hc_narrow_rms_broadcast", "qmoe_fc1_double_buffer"),
        ("build_qmoe_fc1_double_buffer_v3 import", "build_qmoe_fc1_double_buffer import"),
        ("hc_separate_original_outputs_supported", "qmoe_fc1_separate_original_outputs_supported"),
    ):
        source = source.replace(before, after)
    return source


def model_source():
    source = adapt((ROOT / "run_hc_narrow_rms_broadcast_model.py").read_text())
    start = source.index("def verify_profile(")
    stop = source.index("def main():", start)
    source = source[:start] + '''def verify_profile(counts, mode):
    selected = "QmoeFC1SwiGLUDoubleBufferF16" if mode == "candidate" else "QmoeFC1SwiGLUBitcastF16"
    original = "QmoeFC1SwiGLUBitcastF16" if mode == "candidate" else "QmoeFC1SwiGLUDoubleBufferF16"
    assert counts[selected] == 48 and counts[original] == 0, counts
    assert counts["QmoeFC2FinalMixBitcastF16"] == 48, counts
    assert counts["HCPostNormProjectionF16"] == 94, counts
    assert counts["MatMulNBitsBlock32F16"] == 159, counts
    assert counts["DenseColumns8DoubleBuffer"] == 0, counts
    assert counts["DynamicSparseAttentionQueryCache"] == 12, counts
    assert counts["DynamicSparseAttention"] == 0, counts


''' + source[stop:]
    for before, after in (
        ('native_proof["fixtures"] == 36 and native_proof["rounds_per_fixture"] == 128',
         'native_proof["fixtures"] == 23 and native_proof["rounds_per_fixture"] == 32'),
        ('native_proof["comparisons_per_complete_phase"] == 607143936',
         'native_proof["comparisons_per_complete_phase"] == 9748480'),
        ('len(native_proof["captured_sha256"]) == 36', 'len(native_proof["captured_sha256"]) == 23'),
        ('row["instances"] == 12 and row["binding_hits"] > 0', 'row["instances"] == 11 and row["hits"] > 0'),
    ):
        source = replace_once(source, before, after)
    return source


def standard_source():
    source = adapt((ROOT / "run_hc_narrow_rms_broadcast_cpu_local8_standard.py").read_text())
    source = replace_once(source,
        'assert gpu["result"]["program_counts"]["HCPostNormProjectionNarrowRmsBroadcastF16"] == 94',
        'assert gpu["result"]["program_counts"]["QmoeFC1SwiGLUDoubleBufferF16"] == 48')
    return replace_once(source,
        'assert gpu["result"]["program_counts"].get("HCPostNormProjectionF16", 0) == 0',
        'assert gpu["result"]["program_counts"].get("QmoeFC1SwiGLUBitcastF16", 0) == 0\n'
        '    assert gpu["result"]["program_counts"]["HCPostNormProjectionF16"] == 94\n'
        '    assert gpu["result"]["program_counts"]["QmoeFC2FinalMixBitcastF16"] == 48')


def screen_source():
    return adapt((ROOT / "run_hc_narrow_rms_broadcast_cpu_local8_screen.py").read_text())


def invalid_source():
    source = adapt((ROOT / "run_hc_narrow_rms_broadcast_invalid.py").read_text())
    return replace_once(source,
        '    fixtures = sorted(Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm-projection/fixtures-v1").glob("*.onnx"))\n'
        '    if len(fixtures) != 36:',
        '    parent = json.loads((ROOT / "expert-bitcast-unpack-v2-native-gpu7-20261029.qualification.json").read_text())\n'
        '    fixtures = [Path("/home/sayanshaw/repos") / name for name in parent["fixture_sha256"]]\n'
        '    if len(fixtures) != 23:')


if __name__ == "__main__":
    for name, source in (
        ("run_qmoe_fc1_double_buffer_model.py", model_source()),
        ("run_qmoe_fc1_double_buffer_cpu_local8_standard.py", standard_source()),
        ("run_qmoe_fc1_double_buffer_cpu_local8_screen.py", screen_source()),
        ("run_qmoe_fc1_double_buffer_invalid.py", invalid_source()),
    ):
        ast.parse(source)
        with (ROOT / name).open("x") as stream:
            stream.write(source)
        print(name)

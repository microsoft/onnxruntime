from pathlib import Path

from build_int8_lm_head_full_a import FLAG, NAME
from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent
BENCHMARK = ROOT / "benchmark_webgpu_int4_int8_head.py"


def benchmark_source():
    return replace_once((ROOT / "benchmark_webgpu_int4.py").read_text(),
        '                "ORT_WEBGPU_QMOE_GROUPED_PREFILL",\n',
        '                "ORT_WEBGPU_QMOE_GROUPED_PREFILL",\n'
        f'                "{FLAG}",\n')


def benchmark_command(command):
    command = command.copy()
    old = "/evidence/webgpu-perf-oct20/benchmark_webgpu_int4.py"
    assert command[-1].count(old) == 1
    command[-1] = command[-1].replace(old, "/evidence/webgpu-perf-oct20/" + BENCHMARK.name)
    return command


def adapt(source):
    return source.replace("hc_wide_rms_broadcast", "int8_lm_head_full_a").replace(
        "hc-wide-rms-broadcast", "int8-lm-head-full-a")


def model_source():
    source = adapt((ROOT / "run_hc_wide_rms_broadcast_model.py").read_text())
    source = replace_once(source, 'ROOT / "int8-lm-head-full-a-v1-native-outcome.json"',
                          'ROOT / "int8-lm-head-full-a-v4-native-outcome.json"')
    start = source.index('    assert native_proof["status"]')
    stop = source.index("    parent = json.loads(parent_path.read_text())", start)
    source = source[:start] + '''    assert native_proof["status"] == "int8_head_separate_original_outputs_supported"
    assert [row["phase"] for row in native_proof["records"]] == ["capture", "default", "candidate"]
    assert all(row["exit_code"] == 0 for row in native_proof["records"])
    assert native_proof["independent_original_provider_captures"]
    assert native_proof["fixtures"] == 13 and len(native_proof["captured_sha256"]) == 13
    assert native_proof["comparisons_per_complete_phase"] == 15906816
    assert native_proof["actual_model_int8_weights"] and native_proof["float32_shape_fallbacks_tested"]
    assert len(native_proof["coverage"]) == 3
    assert invalid["status"] == "fresh_invalid_int8_lm_head_full_a_selector_rejected"
    for proof in (native_proof, invalid):
        for path, digest in proof["source_and_evidence_sha256"].items():
            assert sha(Path(path)) == digest, path
''' + source[stop:]
    start = source.index("def verify_profile(counts, mode):")
    stop = source.index("\ndef main():", start)
    source = source[:start] + f'''def verify_profile(counts, mode):
    selected = "{NAME}" if mode == "candidate" else "MatMulNBits"
    original = "MatMulNBits" if mode == "candidate" else "{NAME}"
    assert counts[selected] == 1 and counts[original] == 0, counts
    assert counts["HCPostNormProjectionF16"] == 94, counts
    assert counts["MatMulNBitsBlock32F16"] == 159, counts
    assert counts["DenseColumns8DoubleBuffer"] == 0, counts
    assert counts["DynamicSparseAttentionQueryCache"] == 12, counts
    assert counts["DynamicSparseAttention"] == 0, counts

''' + source[stop:]
    source = replace_once(source, "from build_int8_lm_head_full_a import FLAG",
        "from build_int8_lm_head_full_a import FLAG\n"
        "from prepare_int8_lm_head_full_a_model import benchmark_command")
    source = replace_once(source, "    command = standard_command(template, provider, prefix, context, full)",
        "    command = benchmark_command(standard_command(template, provider, prefix, context, full))")
    source = source.replace('ROOT / "prepare_int8_lm_head_full_a_proofs.py"',
                            'ROOT / "prepare_int8_lm_head_full_a_model.py"')
    return source.replace('ROOT / "benchmark_webgpu_int4.py"', f'ROOT / "{BENCHMARK.name}"')


def cpu_source():
    source = adapt((ROOT / "run_hc_wide_rms_broadcast_cpu_local8_standard.py").read_text())
    source = replace_once(source, "from build_int8_lm_head_full_a import FLAG",
        "from build_int8_lm_head_full_a import FLAG\n"
        "from prepare_int8_lm_head_full_a_model import benchmark_command")
    source = replace_once(source, "    command = standard_command(template, provider, prefix, context, full)",
        "    command = benchmark_command(standard_command(template, provider, prefix, context, full))")
    source = replace_once(source,
        'assert gpu["result"]["program_counts"]["HCPostNormProjectionWideRmsBroadcastF16"] == 94',
        f'assert gpu["result"]["program_counts"]["{NAME}"] == 1')
    source = replace_once(source,
        'assert gpu["result"]["program_counts"].get("HCPostNormProjectionF16", 0) == 0',
        'assert gpu["result"]["program_counts"].get("MatMulNBits", 0) == 0\n'
        '    assert gpu["result"]["program_counts"]["HCPostNormProjectionF16"] == 94')
    return source.replace('ROOT / "benchmark_webgpu_int4.py"', f'ROOT / "{BENCHMARK.name}"')


def screen_source():
    return adapt((ROOT / "run_hc_wide_rms_broadcast_cpu_local8_screen.py").read_text())


if __name__ == "__main__":
    for name, source in (
        (BENCHMARK.name, benchmark_source()),
        ("run_int8_lm_head_full_a_model.py", model_source()),
        ("run_int8_lm_head_full_a_cpu_local8_standard.py", cpu_source()),
        ("run_int8_lm_head_full_a_cpu_local8_screen.py", screen_source()),
    ):
        target = ROOT / name
        with target.open("x") as stream:
            stream.write(source)
        print(target)

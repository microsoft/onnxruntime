from pathlib import Path

from build_dsa_four_subgroups import FLAG, NAME
from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent
BENCHMARK = ROOT / "benchmark_webgpu_int4_dsa_four_subgroups.py"


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
    return source.replace("hc_wide_rms_broadcast", "dsa_four_subgroups").replace(
        "hc-wide-rms-broadcast", "dsa-four-subgroups")


def controller_source(kind):
    source = adapt((ROOT / f"run_hc_wide_rms_broadcast_{kind}.py").read_text())
    if kind in ("model", "cpu_local8_standard"):
        source = replace_once(source, "from build_dsa_four_subgroups import FLAG",
            "from build_dsa_four_subgroups import FLAG\n"
            "from prepare_dsa_four_subgroups_model import benchmark_command")
        source = replace_once(source, "    command = standard_command(template, provider, prefix, context, full)",
            "    command = benchmark_command(standard_command(template, provider, prefix, context, full))")
        source = source.replace('ROOT / "benchmark_webgpu_int4.py"', f'ROOT / "{BENCHMARK.name}"')
    if kind == "model":
        start = source.index('    assert native_proof["status"]')
        stop = source.index("    parent = json.loads(parent_path.read_text())", start)
        source = source[:start] + '''    assert native_proof["status"] == "dsa_four_subgroups_independent_original_outputs_supported"
    assert len(native_proof["records"]) == len(native_proof["coverage"]) == 6
    assert [row["phase"] for row in native_proof["records"]] == ["capture", "default", "candidate"] * 2
    assert all(row["exit_code"] == 0 for row in native_proof["records"])
    assert native_proof["fixtures"] == 27 and native_proof["fixed_fixture_repeats"] == 9
    assert len(native_proof["captured_sha256"]) == 54
    assert native_proof["comparisons_per_complete_phase"] == 2435328
    for row in native_proof["coverage"]:
        counts = row["program_counts"]
        assert counts["DynamicSparseAttention"] == 225
        assert counts["DynamicSparseAttentionMerge"] == 63
        assert counts.get("DynamicSparseAttentionFourSubgroups", 0) == (18 if row["phase"] == "candidate" else 0)
        assert counts.get("DynamicSparseAttentionQueryCache", 0) == (0 if row["phase"] == "candidate" else 18)
    assert invalid["status"] == "fresh_invalid_dsa_four_subgroups_selector_rejected"
    for proof in (native_proof, invalid):
        for path, digest in proof["source_and_evidence_sha256"].items():
            assert sha(Path(path)) == digest, path
''' + source[stop:]
        start = source.index("def verify_profile(counts, mode):")
        stop = source.index("\ndef main():", start)
        source = source[:start] + f'''def verify_profile(counts, mode):
    selected = "{NAME}" if mode == "candidate" else "DynamicSparseAttentionQueryCache"
    original = "DynamicSparseAttentionQueryCache" if mode == "candidate" else "{NAME}"
    assert counts[selected] == 12 and counts[original] == 0, counts
    assert counts["DynamicSparseAttention"] == 0, counts
    assert counts["HCPostNormProjectionF16"] == 94, counts
    assert counts["MatMulNBitsBlock32F16"] == 159, counts
    assert counts["DenseColumns8DoubleBuffer"] == 0, counts
    assert counts["MatMulNBits"] == 1, counts

''' + source[stop:]
        source = source.replace('ROOT / "prepare_dsa_four_subgroups_proofs.py"',
                                'ROOT / "prepare_dsa_four_subgroups_model.py"')
    if kind == "cpu_local8_standard":
        source = replace_once(source,
            'assert gpu["result"]["program_counts"]["HCPostNormProjectionWideRmsBroadcastF16"] == 94',
            f'assert gpu["result"]["program_counts"]["{NAME}"] == 12')
        source = replace_once(source,
            'assert gpu["result"]["program_counts"].get("HCPostNormProjectionF16", 0) == 0',
            'assert gpu["result"]["program_counts"].get("DynamicSparseAttentionQueryCache", 0) == 0\n'
            '    assert gpu["result"]["program_counts"]["HCPostNormProjectionF16"] == 94')
    return source


if __name__ == "__main__":
    outputs = [(BENCHMARK.name, benchmark_source())]
    outputs.extend((f"run_dsa_four_subgroups_{kind}.py", controller_source(kind))
                   for kind in ("model", "cpu_local8_standard", "cpu_local8_screen"))
    for name, source in outputs:
        with (ROOT / name).open("x") as stream:
            stream.write(source)
        print(ROOT / name)

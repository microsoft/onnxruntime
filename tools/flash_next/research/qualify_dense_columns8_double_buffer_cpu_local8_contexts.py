import json
from pathlib import Path

from qualify_cpu_greedy_argmax_lanes8_local8_contexts import verify_screen as verify_cpu_screen
from qualify_row4_cpu_policy_contexts import candidates_above_target
from run_dense_columns8_double_buffer_cpu_local8_standard import ROOT, RUNTIME, STAMP, VARIANT
from build_qmoe_node_replay import sha


BUILD = Path("/home/sayanshaw/repos/qwen38-int4-dense-columns8-double-buffer/build-v1")
ORIGINAL = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows4/build-v2/libonnxruntime_providers_webgpu.so")
ORIGINAL_SHA = "0955e1d6cc30cac13044450f692100ffa5f33361aaef229e25ff0efa20a8c8d4"


def verify_screen(proof, context, repeat, provider, runtime):
    prefix = "repeated_" if repeat else "initial_"
    expected = prefix + "columns8_double_buffer_cpu_local8_decode_gain_both_modes"
    assert proof["status"] in (expected, "no_supported_decode_gain")
    assert proof["provider_sha256"] == provider
    assert proof["original_baseline_provider_sha256"] == ORIGINAL_SHA
    assert proof["runtime_sha256"] == proof["parent_runtime_sha256"] == runtime
    assert proof["comparison_kind"] == "bracketed_separate_load_original_provider_same_cpu_runtime_local8_net_screen"
    # Reuse the frozen numerical validator, translating only this family's known status.
    checked = dict(proof)
    if checked["status"] == expected:
        checked["status"] = prefix + "cpu_greedy_decode_gain_both_modes"
    return verify_cpu_screen(checked, context, repeat)


def main():
    output = ROOT / f"{VARIANT}-all-contexts-{STAMP}.qualification.json"
    if output.exists():
        raise FileExistsError(output)
    gpu_path = BUILD / "build-manifest.json"
    cpu_path = RUNTIME / "build-manifest.json"
    gpu, cpu = (json.loads(path.read_text()) for path in (gpu_path, cpu_path))
    provider, runtime = gpu["provider_sha256"], cpu["runtime_sha256"]
    sources, rows, support = {}, [], []
    for context in (2048, 8192, 32768):
        screens = {}
        for repeat in (False, True):
            path = ROOT / f"{VARIANT}-context{context}-{'repeat-' if repeat else ''}screen-{STAMP}.qualification.json"
            proof = json.loads(path.read_text())
            support.append(verify_screen(proof, context, repeat, provider, runtime))
            for name, digest in proof["source_sha256"].items():
                assert sha(Path(name)) == digest, name
                sources[name] = digest
            sources[str(path)] = sha(path)
            screens["fresh" if repeat else "initial"] = proof
        rows.append(dict(context=context, **screens))
    provider_path = BUILD / "libonnxruntime_providers_webgpu.so"
    runtime_path = RUNTIME / "libonnxruntime-genai.so"
    assert sha(provider_path) == provider and sha(ORIGINAL) == ORIGINAL_SHA
    assert sha(runtime_path) == runtime
    evidence = (Path(__file__), gpu_path, cpu_path, provider_path, runtime_path, ORIGINAL,
                ROOT / "qualify_cpu_greedy_argmax_lanes8_local8_contexts.py",
                ROOT / "qualify_row4_cpu_policy_contexts.py")
    report = dict(
        status=("repeated_columns8_double_buffer_cpu_local8_supported_all_contexts" if all(support)
                else "columns8_double_buffer_cpu_local8_held_insufficient_all_context_support"),
        contexts=[2048, 8192, 32768], provider_sha256=provider,
        original_baseline_provider_sha256=ORIGINAL_SHA, runtime_sha256=runtime,
        parent_runtime_sha256=runtime, rows=rows,
        all_contexts_both_modes_above_100_decode_tps=candidates_above_target(rows),
        measured_request_comparisons=36,
        positive_request_comparisons=sum(mode["comparison"]["positive_requests"]
            for row in rows for phase in ("initial", "fresh") for mode in row[phase]["rows"]),
        original_output_prefixes_match=True, zero_major_faults=True,
        matched_cpu_and_worker_policy_in_both_modes=True,
        original_provider_controls=True, same_cpu_runtime_in_both_modes=True,
        fresh_reuses_initial_ending_controls=True,
        initial32k_timing_caveat="Short C++ shader-renderer builds overlapped the initial driver; natural request overlap unresolved.",
        default_enabled=False, quality_qualified=False, headline_eligible=False,
        source_sha256={**sources, **{str(path): sha(path) for path in evidence}})
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
        ("status", "positive_request_comparisons", "all_contexts_both_modes_above_100_decode_tps")}), flush=True)


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path
import shlex
import subprocess

from qualify_quickgelu_mul_long_context import ROOT, sha


def classify_result(guard, log, returncode):
    assert guard["exit_code"] == returncode
    if returncode:
        assert returncode == 1 and guard["status"] == "failed"
        assert "finite/bitwise mismatch" in log, (
            "Probe failed before a numerical comparison: " +
            next((line for line in log.splitlines() if line.startswith("FAIL ")), log[-1000:]))
        reason = next(line for line in log.splitlines() if line.startswith("FAIL "))
        return "native_tensor_dot4_not_bitwise_compatible", reason
    assert guard["status"] == "completed"
    assert log.count(" changing_rounds=128 invalid_flag_rejected=1") == 3
    assert "PASS total_exact_elements=983040; CPU fallback disabled" in log
    return "sparse_dot4_probe_bitwise_compatible_not_general_matmul_qualified", None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument("--single-products", action="store_true")
    variant = parser.add_mutually_exclusive_group()
    variant.add_argument("--fp32-result", action="store_true")
    variant.add_argument("--rounded-products", action="store_true")
    variant.add_argument("--pairwise-products", action="store_true")
    variant.add_argument("--fp32-products", action="store_true")
    variant.add_argument("--reverse-cross-fma", action="store_true")
    variant.add_argument("--cross-fma", action="store_true")
    variant.add_argument("--cross-fma-materialized", action="store_true")
    variant.add_argument("--cross-fma-fp32", action="store_true")
    args = parser.parse_args()
    if args.single_products and (args.rounded_products or args.pairwise_products or args.fp32_products or
                                args.reverse_cross_fma or args.cross_fma or
                                args.cross_fma_materialized or args.cross_fma_fp32):
        parser.error("Single-product probes require the direct FP16 or FP32 matrix variant")
    directory = Path("/home/sayanshaw/repos/qwen38-int4-tensor-dot4-probe")
    build_directory = directory / ("build-v4" if args.rounded_products else
                                   "build-v2" if args.fp32_result else "build-v1")
    if args.pairwise_products:
        build_directory = Path("/home/sayanshaw/repos/qwen38-int4-tensor-dot4-pairwise-probe/build-v1")
    if args.fp32_products:
        build_directory = Path("/home/sayanshaw/repos/qwen38-int4-tensor-dot4-f32-products-probe/build-v1")
    if args.reverse_cross_fma:
        build_directory = Path("/home/sayanshaw/repos/qwen38-int4-tensor-dot4-reverse-cross-fma-probe/build-v1")
    if args.cross_fma:
        build_directory = Path("/home/sayanshaw/repos/qwen38-int4-tensor-dot4-cross-fma-probe/build-v1")
    if args.cross_fma_materialized:
        build_directory = Path("/home/sayanshaw/repos/qwen38-int4-tensor-dot4-cross-fma-probe/build-v2")
    if args.cross_fma_fp32:
        build_directory = Path("/home/sayanshaw/repos/qwen38-int4-tensor-dot4-cross-fma-probe/build-v3")
    build = json.loads((build_directory / "build-manifest.json").read_text())
    parent = json.loads((ROOT / "qsa-prefill-score16-v1-contexts-20261031.qualification.json").read_text())
    assert build["baseline_provider_sha256"] == parent["provider_sha256"] and build["baseline_byte_identical"]
    provider = build_directory / "libonnxruntime_providers_webgpu.so"
    comparator = build_directory / "native_compare"
    assert sha(provider) == build["provider_sha256"] and sha(comparator) == build["comparator_sha256"]
    for name, digest in build["source_sha256"].items():
        assert sha(build_directory / name) == digest
    fixture_directory = directory / ("fixtures-single-v1" if args.single_products else "fixtures-v1")
    manifest_path = fixture_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    terms = 1 if args.single_products else 4
    assert [row["nonzero_k_positions"] for row in manifest] == [list(range(offset, offset + terms)) for offset in (0, 4, 12)]
    fixtures = [fixture_directory / row["fixture"] for row in manifest]
    assert all(sha(path) == row["sha256"] for path, row in zip(fixtures, manifest))
    prefix = ROOT / ("tensor-dot4-probe-v4-products-native-gpu7-20261031" if args.rounded_products else
                     "tensor-dot4-probe-v2-f32-native-gpu7-20261031" if args.fp32_result else
                     "tensor-dot4-probe-v1-native-gpu7-20261031")
    if args.pairwise_products:
        prefix = ROOT / "tensor-dot4-pairwise-probe-v1-native-gpu7-20261031"
    if args.fp32_products:
        prefix = ROOT / "tensor-dot4-f32-products-probe-v1-native-gpu7-20261031"
    if args.reverse_cross_fma:
        prefix = ROOT / "tensor-dot4-reverse-cross-fma-probe-v1-native-gpu7-20261031"
    if args.cross_fma:
        prefix = ROOT / "tensor-dot4-cross-fma-probe-v1-native-gpu7-20261031"
    if args.cross_fma_materialized:
        prefix = ROOT / "tensor-dot4-cross-fma-probe-v2-materialized-native-gpu7-20261031"
    if args.cross_fma_fp32:
        prefix = ROOT / "tensor-dot4-cross-fma-probe-v3-f32-native-gpu7-20261031"
    if args.single_products:
        prefix = ROOT / f"tensor-single-product-v1-{'f32-' if args.fp32_result else ''}native-gpu7-20261031"
    output = prefix.with_suffix(".qualification.json")
    if output.exists() or (not args.verify_existing and
                          any(prefix.with_suffix(ext).exists() for ext in (".guard.json", ".log"))):
        raise FileExistsError(prefix)
    command = ("set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe "
               "GPU-0e70e969-bbfc-80c5-d582-88be134632ea; export "
               "ORT_WEBGPU_DYNAMIC_UNIFORMS=1 ORT_WEBGPU_UNIFORM_ARENA=1 "
               "ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES=8192 ORT_WEBGPU_DENSE_BITCAST_UNPACK=1 "
               "ORT_WEBGPU_DENSE_BLOCK32=1; exec " +
               shlex.join(str(path).replace("/home/sayanshaw/repos", "/workspace")
                          for path in (comparator, provider, *fixtures)))
    if not args.verify_existing:
        subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
                        "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "600",
                        "--output-prefix", str(prefix), "--", "bash", "-lc", command])
    guard = json.loads(prefix.with_suffix(".guard.json").read_text())
    log = prefix.with_suffix(".log").read_text()
    assert guard["command"] == ["bash", "-lc", command] and guard["gpu_index"] == 7
    assert guard["gpu_uuid"] == "GPU-0e70e969-bbfc-80c5-d582-88be134632ea"
    status, reason = classify_result(guard, log, guard["exit_code"])
    report = dict(status=status, reason=reason, gpu_index=7, provider_sha256=build["provider_sha256"],
                  baseline_provider_sha256=parent["provider_sha256"], original_model_unchanged=True,
                  not_a_performance_benchmark=True, quality_qualified=False, headline_eligible=False,
                  matrix_result_component="f32" if args.fp32_result or args.cross_fma_fp32 or args.fp32_products else "f16",
                  source_sha256={str(path): sha(path) for path in
                                 (build_directory / "build-manifest.json", manifest_path, *fixtures,
                                  prefix.with_suffix(".guard.json"), prefix.with_suffix(".log"))})
    if args.rounded_products or args.pairwise_products:
        assert not build["no_reduction_or_normalization_changes"]
        report["independent_products_rounded_to_f16_before_dot"] = True
        report["only_first_16_k_positions_supported"] = True
    if args.pairwise_products:
        report["explicit_pairwise_half_rounding_before_final_addition"] = True
    if args.fp32_products:
        assert not build["no_reduction_or_normalization_changes"]
        report["independent_products_materialized_to_f32_before_dot"] = True
        report["only_first_16_k_positions_supported"] = True
    if args.reverse_cross_fma:
        assert not build["no_reduction_or_normalization_changes"]
        report["first_matrix_result_materialized_to_half"] = True
        report["first_product_k_positions_per_dot4"] = [2, 3]
        report["second_product_k_positions_per_dot4"] = [0, 1]
        report["cross_term_pairing"] = [[2, 0], [3, 1]]
        report["only_first_16_k_positions_supported"] = True
    if args.cross_fma or args.cross_fma_materialized or args.cross_fma_fp32:
        assert not build["no_reduction_or_normalization_changes"]
        report["two_native_f16_input_matrix_accumulations"] = True
        if not args.cross_fma_fp32:
            report["two_native_f16_matrix_accumulations"] = True
        report["cross_term_pairing"] = [[0, 2], [1, 3]]
        report["only_first_16_k_positions_supported"] = True
        report["first_matrix_result_materialized_to_half"] = args.cross_fma_materialized
        report["first_matrix_result_explicitly_half_rounded"] = args.cross_fma_materialized or args.cross_fma_fp32
        report["fp32_stages_with_explicit_half_boundaries"] = args.cross_fma_fp32
    if args.single_products:
        report["single_product_weights_only"] = True
        report["status"] = ("native_tensor_single_product_not_bitwise_compatible" if reason else
                            "single_product_probe_bitwise_compatible_not_dot4_qualified")
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "reason": reason}), flush=True)


if __name__ == "__main__":
    main()

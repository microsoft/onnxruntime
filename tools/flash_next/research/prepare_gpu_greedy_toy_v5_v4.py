from pathlib import Path

from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent
STATUS = "toy_runtime_ids_non_nan_score_bits_nan_positions_cpu_authority_exact"


if __name__ == "__main__":
    source = (ROOT / "gpu_greedy_toy_v5_v3.py").read_text()
    source = replace_once(source, "    checks, score_checks = [], 0",
                          "    checks, score_checks, nan_payload_differences = [], 0, 0")
    source = replace_once(source,
        "            np.testing.assert_array_equal(scores.view(np.uint32), api_expected.view(np.uint32))",
        "            actual_nan, expected_nan = np.isnan(scores), np.isnan(api_expected)\n"
        "            np.testing.assert_array_equal(actual_nan, expected_nan)\n"
        "            actual_bits, expected_bits = scores.view(np.uint32), api_expected.view(np.uint32)\n"
        "            np.testing.assert_array_equal(actual_bits[~expected_nan], expected_bits[~expected_nan])\n"
        "            differences = int(np.count_nonzero(actual_bits[expected_nan] != expected_bits[expected_nan]))\n"
        "            if dtype == np.float32 and differences:\n"
        '                raise RuntimeError("F32 scores changed NaN payloads without a conversion")\n'
        "            nan_payload_differences += differences")
    source = source.replace("toy_runtime_ids_scores_cpu_authority_exact", STATUS)
    source = replace_once(source, "                  public_get_logits_computes_next_step_when_needed=True,",
        "                  public_get_logits_computes_next_step_when_needed=True,\n"
        "                  non_nan_score_bits_exact=True, nan_positions_exact=True,\n"
        "                  f16_nan_payload_cpu_identity_qualified=False,\n"
        "                  nan_payload_differences=nan_payload_differences,")
    with (ROOT / "gpu_greedy_toy_v5_v4.py").open("x") as stream:
        stream.write(source)
    source = (ROOT / "run_gpu_greedy_toy_v5_native.py").read_text()
    source = source.replace('"gpu-greedy-argmax-v5-toy-native-outcome.json"',
                            '"gpu-greedy-argmax-v5-toy-native-v2-outcome.json"')
    source = source.replace("gpu_greedy_toy_v5_v3.py", "gpu_greedy_toy_v5_v4.py")
    source = source.replace('f"gpu-greedy-argmax-v5-toy-native-flag', 'f"gpu-greedy-argmax-v5-toy-native-v2-flag')
    source = replace_once(source, '        command = [part.replace(original_prefix.name, prefix.name) for part in guard["command"]]',
        '        command = [part.replace(original_prefix.name, prefix.name).replace(\n'
        '            "gpu_greedy_toy_v5_v3.py", "gpu_greedy_toy_v5_v4.py") for part in guard["command"]]')
    # The pilot is F32-only and retains its original exact-bit status.
    source = replace_once(source, 'result["status"] != "toy_runtime_ids_scores_cpu_authority_exact"',
                          f'result["status"] != "{STATUS}"')
    source = replace_once(source, '                           gpu_peak_gib=state["peak_gpu_used_bytes"] / 1024**3))',
        '                           gpu_peak_gib=state["peak_gpu_used_bytes"] / 1024**3,\n'
        '                           non_nan_score_bits_exact=result["non_nan_score_bits_exact"],\n'
        '                           nan_positions_exact=result["nan_positions_exact"],\n'
        '                           nan_payload_differences=result["nan_payload_differences"]))')
    source = source.replace("coalesced_helper_toy_ids_api_penalties_sampling_exact",
                            "coalesced_helper_toy_ids_api_numeric_scores_penalties_sampling_exact")
    source = replace_once(source, "        model_eligible=False, performance_qualified=False,",
        "        f16_nan_payload_cpu_identity_qualified=False,\n"
        "        model_eligible=False, performance_qualified=False,")
    with (ROOT / "run_gpu_greedy_toy_v5_native_v2.py").open("x") as stream:
        stream.write(source)

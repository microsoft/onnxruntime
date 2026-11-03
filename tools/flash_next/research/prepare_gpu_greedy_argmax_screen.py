from pathlib import Path

from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent


if __name__ == "__main__":
    source = (ROOT / "run_cpu_greedy_argmax_avx512_screen.py").read_text()
    source = replace_once(source,
        "from run_cpu_greedy_argmax_avx512_standard import BUILD, FLAG, STAMP, VARIANT, prefix_for, verify_proof, verify_workers",
        "from run_gpu_greedy_argmax_standard import BUILD, FLAG, VARIANT, prefix_for, verify_proof, verify_workers")
    source = source.replace("screen-{STAMP}", "screen")
    source = source.replace("run_cpu_greedy_argmax_avx512_standard.py", "run_gpu_greedy_argmax_standard.py")
    source = source.replace('"avx512"', '"gpu"')
    source = source.replace("cpu_greedy_avx512_original_prefixes_qualified", "gpu_greedy_original_prefixes_qualified")
    source = source.replace("cpu_greedy_decode_gain", "gpu_greedy_decode_gain")
    source = source.replace("same_runtime_both_sides", "same_runtime_and_provider_both_sides")
    source = source.replace("control_is_lanes8_with_avx512_disabled", "control_is_lanes8_with_gpu_greedy_disabled")
    with (ROOT / "run_gpu_greedy_argmax_screen.py").open("x") as stream:
        stream.write(source)

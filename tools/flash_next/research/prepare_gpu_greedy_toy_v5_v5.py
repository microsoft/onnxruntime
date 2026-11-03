from pathlib import Path

from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent


if __name__ == "__main__":
    source = (ROOT / "gpu_greedy_toy_v5_v4.py").read_text()
    source = replace_once(source, "do_sample=True, top_k=1, top_p=1.0",
                          "do_sample=True, top_k=2, top_p=1.0, random_seed=42")
    source = replace_once(source,
        "        if int(gen.get_next_tokens()[0]) != 29:",
        "        sampled = int(gen.get_next_tokens()[0])\n"
        "        if sampled not in (5, 29):")
    source = replace_once(source, "sampling_top1=True, token_id=29", "sampling_top2_seed42=True, token_id=sampled")
    with (ROOT / "gpu_greedy_toy_v5_v5.py").open("x") as stream:
        stream.write(source)
    source = (ROOT / "run_gpu_greedy_toy_v5_native_v2.py").read_text()
    source = source.replace("toy-native-v2", "toy-native-v3")
    source = source.replace("gpu_greedy_toy_v5_v4.py", "gpu_greedy_toy_v5_v5.py")
    source = replace_once(source, "    phases = []", "    phases = []\n    baseline_checks = None")
    source = replace_once(source, '        if len(result["checks"]) != 108',
        "        if flag == 0:\n"
        '            baseline_checks = result["checks"]\n'
        '        elif result["checks"] != baseline_checks:\n'
        '            raise RuntimeError("Fused helper changed fixed-seed CPU sampling or validated IDs")\n'
        '        if len(result["checks"]) != 108')
    with (ROOT / "run_gpu_greedy_toy_v5_native_v3.py").open("x") as stream:
        stream.write(source)

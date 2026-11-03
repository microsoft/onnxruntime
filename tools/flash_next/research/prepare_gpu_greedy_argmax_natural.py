from pathlib import Path

from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent


if __name__ == "__main__":
    source = (ROOT / "benchmark_gpu_greedy_argmax.py").read_text()
    source = replace_once(source, "    generator.append_tokens(prompt)\n    prefill_finished",
        '    generator.append_tokens(prompt)\n'
        '    generator.get_output("logits")\n'
        '    prefill_finished')
    source = replace_once(source, '"append_tokens_wall_including_logits_ready"',
                          '"append_tokens_plus_raw_output_cpu_synchronization"')
    with (ROOT / "benchmark_gpu_greedy_argmax_natural.py").open("x") as stream:
        stream.write(source)

from pathlib import Path

from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent


def write_new(name, source):
    with (ROOT / name).open("x") as stream:
        stream.write(source)


if __name__ == "__main__":
    source = (ROOT / "benchmark_gpu_greedy_argmax.py").read_text()
    source = replace_once(source, "    prefill_finished = time.perf_counter()\n", "    prefill_finished = None\n")
    source = replace_once(source, "        generator.generate_next_token()\n",
        "        generator.generate_next_token()\n"
        "        if index == 0:\n"
        "            prefill_finished = time.perf_counter()\n")
    source = replace_once(source, "    if usage_first is None:\n",
                          "    if usage_first is None or prefill_finished is None:\n")
    source = replace_once(source, '"append_tokens_wall_including_logits_ready"',
                          '"append_tokens_through_first_token_selection"')
    write_new("benchmark_gpu_greedy_argmax_ready.py", source)
    source = (ROOT / "run_gpu_greedy_argmax_standard.py").read_text()
    source = replace_once(source, 'VARIANT = "gpu-greedy-argmax-local8-v4"',
                          'VARIANT = "gpu-greedy-argmax-local8-v4-ready1"')
    source = source.replace("benchmark_gpu_greedy_argmax_natural.py", "benchmark_gpu_greedy_argmax_ready.py")
    source = source.replace("append_tokens_plus_raw_output_cpu_synchronization",
                            "append_tokens_through_first_token_selection")
    source = source.replace("prepare_gpu_greedy_argmax_natural.py", "prepare_gpu_greedy_argmax_ready.py")
    source = source.replace("prefill_includes_raw_output_cpu_synchronization", "prefill_includes_first_token_selection")
    write_new("run_gpu_greedy_argmax_ready_standard.py", source)
    source = (ROOT / "run_gpu_greedy_argmax_screen.py").read_text()
    source = source.replace("run_gpu_greedy_argmax_standard", "run_gpu_greedy_argmax_ready_standard")
    write_new("run_gpu_greedy_argmax_ready_screen.py", source)

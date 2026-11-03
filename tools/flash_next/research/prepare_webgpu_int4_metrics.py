from pathlib import Path

from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent
BENCHMARK = ROOT / "benchmark_webgpu_int4_metrics.py"
GUARD = ROOT / "shared_activation_metrics_guard.py"


def benchmark_source():
    source = (ROOT / "benchmark_webgpu_int4.py").read_text()
    source = replace_once(source, "    generator.append_tokens(prompt)\n",
                          "    generator.append_tokens(prompt)\n    prefill_finished = time.perf_counter()\n")
    source = replace_once(source, '        "decode_seconds": times[-1] - times[0],',
        '        "prefill_seconds": prefill_finished - start,\n'
        '        "prefill_tps": len(prompt) / (prefill_finished - start),\n'
        '        "prefill_timing_scope": "append_tokens_wall_including_logits_ready",\n'
        '        "decode_seconds": times[-1] - times[0],')
    return replace_once(source, '                "ORT_WEBGPU_QMOE_GROUPED_PREFILL",\n',
        '                "ORT_WEBGPU_QMOE_GROUPED_PREFILL",\n'
        '                "ORT_WEBGPU_DENSE_COLUMNS8_DOUBLE_BUFFER",\n'
        '                "ORT_GENAI_CPU_GREEDY_ARGMAX_LANES8",\n'
        '                "ORT_GENAI_CPU_GREEDY_ARGMAX_AVX512",\n')


def guard_source():
    source = (ROOT / "shared_activation_guard.py").read_text()
    source = "import ctypes\nfrom nvml_monitor import MemoryInfo\n\n" + source
    anchor = "\ndef main():"
    helper = '''
def used_gpu_bytes(monitor, uuid):
    memory = MemoryInfo()
    monitor._check(monitor.library.nvmlDeviceGetMemoryInfo(
        monitor._handle(uuid), ctypes.byref(memory)), "nvmlDeviceGetMemoryInfo")
    return memory.used

'''
    source = replace_once(source, anchor, helper + anchor)
    source = replace_once(source, '    state["status"] = "running"',
        '    state["peak_gpu_used_bytes"] = used_gpu_bytes(monitor, gpu_uuid)\n'
        '    state["baseline_gpu_used_bytes"] = state["peak_gpu_used_bytes"]\n'
        '    state["gpu_memory_samples"] = 1\n'
        '    state["gpu_memory_sampling_interval_seconds"] = 0.5\n'
        '    state["gpu_memory_scope"] = "device_used_container_lifetime_including_load_and_warmup"\n\n'
        '    state["status"] = "running"')
    return replace_once(source, "            while process.poll() is None:",
        "            while process.poll() is None:\n"
        '                state["peak_gpu_used_bytes"] = max(state["peak_gpu_used_bytes"],\n'
        '                                                   used_gpu_bytes(monitor, gpu_uuid))\n'
        '                state["gpu_memory_samples"] += 1')


if __name__ == "__main__":
    for path, source in ((BENCHMARK, benchmark_source()), (GUARD, guard_source())):
        with path.open("x") as stream:
            stream.write(source)
        print(path)

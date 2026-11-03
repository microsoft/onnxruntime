from pathlib import Path

from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent


if __name__ == "__main__":
    source = (ROOT / "benchmark_webgpu_int4_metrics.py").read_text()
    source = replace_once(source, '                "ORT_GENAI_CPU_GREEDY_ARGMAX_AVX512",\n',
        '                "ORT_GENAI_CPU_GREEDY_ARGMAX_AVX512",\n'
        '                "ORT_GENAI_WEBGPU_GREEDY_ARGMAX",\n'
        '                "ORT_GENAI_WEBGPU_GREEDY_ARGMAX_TRACE",\n'
        '                "ORT_WEBGPU_ARGMAX_PARALLEL_BITWISE",\n')
    with (ROOT / "benchmark_gpu_greedy_argmax.py").open("x") as stream:
        stream.write(source)

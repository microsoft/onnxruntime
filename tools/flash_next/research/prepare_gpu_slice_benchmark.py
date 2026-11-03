from pathlib import Path

from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent


if __name__ == "__main__":
    source = (ROOT / "benchmark_gpu_greedy_argmax_ready.py").read_text()
    source = replace_once(source, '                "ORT_GENAI_WEBGPU_GREEDY_ARGMAX_TRACE",',
        '                "ORT_GENAI_WEBGPU_GREEDY_ARGMAX_TRACE",\n'
        '                "ORT_GENAI_WEBGPU_GPU_SLICE_COPY",\n'
        '                "ORT_WEBGPU_SLICE_UINT32_BITCOPY",\n'
        '                "ORT_GENAI_WEBGPU_GREEDY_ARGMAX_FUSED_MIN_LENGTH",')
    with (ROOT / "benchmark_gpu_slice.py").open("x") as stream:
        stream.write(source)

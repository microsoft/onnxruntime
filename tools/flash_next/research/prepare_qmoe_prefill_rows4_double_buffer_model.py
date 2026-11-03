from pathlib import Path

from build_qmoe_node_replay import replace_once
from build_qmoe_prefill_rows4_double_buffer import FLAG


ROOT = Path(__file__).parent
BENCHMARK = ROOT / "benchmark_webgpu_int4_prefill_double_buffer.py"


def benchmark_source():
    return replace_once((ROOT / "benchmark_webgpu_int4.py").read_text(),
        '                "ORT_WEBGPU_QMOE_GROUPED_PREFILL",\n',
        '                "ORT_WEBGPU_QMOE_GROUPED_PREFILL",\n'
        f'                "{FLAG}",\n')


if __name__ == "__main__":
    with BENCHMARK.open("x") as stream:
        stream.write(benchmark_source())
    print(BENCHMARK)

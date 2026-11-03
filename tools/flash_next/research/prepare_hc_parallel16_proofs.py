from pathlib import Path

from build_hc_parallel16 import FLAG, NAME
from build_qmoe_node_replay import replace_once
from prepare_webgpu_int4_metrics import BENCHMARK as METRICS_BENCHMARK


ROOT = Path(__file__).parent
BENCHMARK = ROOT / "benchmark_webgpu_int4_hc_parallel16.py"


def adapt(source):
    source = source.replace("hc_owned_vector", "hc_parallel16")
    source = source.replace("hc-owned-vector", "hc-parallel16")
    source = source.replace("ORT_WEBGPU_HC_OWNED_VECTOR", FLAG)
    source = source.replace("HCPostNormProjectionOwnedVectorF16", NAME)
    source = source.replace("hc-parallel16-v1-native-outcome.json",
                            "hc-parallel16-v1-native-attempt3-outcome.json")
    source = source.replace('STAMP = "20261101"', 'STAMP = "20261103"')
    return source.replace("hc-parallel16-v1-candidate-gpu7-full-profile2048-trial0-20261101",
                          "hc-parallel16-v1-candidate-gpu7-full-profile2048-trial0-20261103")


def benchmark_source():
    return replace_once(METRICS_BENCHMARK.read_text(),
        '                "ORT_WEBGPU_QMOE_GROUPED_PREFILL",\n',
        '                "ORT_WEBGPU_QMOE_GROUPED_PREFILL",\n'
        f'                "{FLAG}",\n')


def benchmark_command(command):
    command = command.copy()
    old = "/evidence/webgpu-perf-oct20/benchmark_webgpu_int4.py"
    assert command[-1].count(old) == 1
    command[-1] = "export ORT_GENAI_CPU_GREEDY_ARGMAX_AVX512=0; " + command[-1].replace(
        old, "/evidence/webgpu-perf-oct20/" + BENCHMARK.name)
    return command


def controller_source(kind):
    source = adapt((ROOT / f"run_hc_owned_vector_{kind}.py").read_text())
    if kind == "invalid":
        source = source.replace("hc-parallel16-v1-invalid-gpu7", "hc-parallel16-v1-invalid-gpu4")
    if kind in ("model", "cpu_local8_standard"):
        source = replace_once(source,
            'str(ROOT / "shared_activation_guard.py")',
            'str(ROOT / "shared_activation_metrics_guard.py")')
    return source


if __name__ == "__main__":
    outputs = [(BENCHMARK, benchmark_source())]
    outputs.extend((ROOT / f"run_hc_parallel16_{kind}.py", controller_source(kind))
                   for kind in ("invalid", "model", "cpu_local8_standard", "cpu_local8_screen"))
    for path, source in outputs:
        with path.open("x") as stream:
            stream.write(source)
        print(path.name)

from pathlib import Path

from build_hc_separate8 import FLAG
from build_qmoe_node_replay import replace_once
from prepare_webgpu_int4_metrics import BENCHMARK as METRICS_BENCHMARK


ROOT = Path(__file__).parent
BENCHMARK = ROOT / "benchmark_webgpu_int4_hc_separate8.py"


def adapt(source):
    source = source.replace("hc_owned_vector", "hc_separate8").replace("hc-owned-vector", "hc-separate8")
    source = source.replace("ORT_WEBGPU_HC_OWNED_VECTOR", FLAG)
    source = source.replace("HCPostNormProjectionOwnedVectorF16", "HCSeparate8ProjectionF16")
    source = source.replace('STAMP = "20261101"', 'STAMP = "20261103"')
    return source.replace("hc-separate8-v1-candidate-gpu7-full-profile2048-trial0-20261101",
                          "hc-separate8-v1-candidate-gpu7-full-profile2048-trial0-20261103")


def benchmark_command(command):
    command = command.copy()
    old = "/evidence/webgpu-perf-oct20/benchmark_webgpu_int4.py"
    assert command[-1].count(old) == 1
    command[-1] = "export ORT_GENAI_CPU_GREEDY_ARGMAX_AVX512=0; " + command[-1].replace(
        old, "/evidence/webgpu-perf-oct20/" + BENCHMARK.name)
    return command


def controller_source(kind):
    source = adapt((ROOT / f"run_hc_owned_vector_{kind}.py").read_text())
    if kind == "model":
        source = replace_once(source,
            '    assert counts[selected] == 94 and counts[original] == 0, counts',
            '    assert counts[selected] == 94 and counts[original] == 0, counts\n'
            '    assert counts["HCSeparate8NormF16"] == (94 if mode == "candidate" else 0), counts')
    if kind == "cpu_local8_standard":
        source = replace_once(source,
            '    assert gpu["result"]["program_counts"]["HCSeparate8ProjectionF16"] == 94',
            '    assert gpu["result"]["program_counts"]["HCSeparate8ProjectionF16"] == 94\n'
            '    assert gpu["result"]["program_counts"]["HCSeparate8NormF16"] == 94')
    if kind in ("model", "cpu_local8_standard"):
        source = replace_once(source, 'str(ROOT / "shared_activation_guard.py")',
                              'str(ROOT / "shared_activation_metrics_guard.py")')
    return source


if __name__ == "__main__":
    benchmark = replace_once(METRICS_BENCHMARK.read_text(),
        '                "ORT_WEBGPU_QMOE_GROUPED_PREFILL",\n',
        '                "ORT_WEBGPU_QMOE_GROUPED_PREFILL",\n'
        f'                "{FLAG}",\n')
    outputs = [(BENCHMARK, benchmark)]
    outputs.extend((ROOT / f"run_hc_separate8_{kind}.py", controller_source(kind))
                   for kind in ("native", "invalid", "model", "cpu_local8_standard", "cpu_local8_screen"))
    for path, source in outputs:
        with path.open("x") as stream:
            stream.write(source)
        print(path.name)

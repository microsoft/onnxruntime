from pathlib import Path

from build_hc_owned_vector import FLAG, NAME
from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent
BENCHMARK = ROOT / "benchmark_webgpu_int4_hc_owned_vector.py"


def adapt(source):
    return source.replace("hc_wide_rms_broadcast", "hc_owned_vector").replace(
        "hc-wide-rms-broadcast", "hc-owned-vector").replace(
        "HCPostNormProjectionWideRmsBroadcastF16", NAME)


def benchmark_source():
    return replace_once((ROOT / "benchmark_webgpu_int4.py").read_text(),
        '                "ORT_WEBGPU_QMOE_GROUPED_PREFILL",\n',
        '                "ORT_WEBGPU_QMOE_GROUPED_PREFILL",\n'
        f'                "{FLAG}",\n')


def benchmark_command(command):
    command = command.copy()
    old = "/evidence/webgpu-perf-oct20/benchmark_webgpu_int4.py"
    assert command[-1].count(old) == 1
    command[-1] = command[-1].replace(old, "/evidence/webgpu-perf-oct20/" + BENCHMARK.name)
    return command


def controller_source(kind):
    source = adapt((ROOT / f"run_hc_wide_rms_broadcast_{kind}.py").read_text())
    if kind in ("model", "cpu_local8_standard"):
        source = replace_once(source, "from build_hc_owned_vector import FLAG",
            "from build_hc_owned_vector import FLAG\n"
            "from prepare_hc_owned_vector_proofs import benchmark_command")
        source = replace_once(source, "    command = standard_command(template, provider, prefix, context, full)",
            "    command = benchmark_command(standard_command(template, provider, prefix, context, full))")
        source = source.replace('ROOT / "benchmark_webgpu_int4.py"', f'ROOT / "{BENCHMARK.name}"')
    if kind == "model":
        source = replace_once(source, '    assert counts["MatMulNBitsBlock32F16"] == 159, counts',
            '    assert counts["MatMulNBitsBlock32F16"] == 159, counts\n'
            '    assert counts["MatMulNBits"] == 1, counts')
    return source


if __name__ == "__main__":
    outputs = [(BENCHMARK.name, benchmark_source())]
    outputs.extend((f"run_hc_owned_vector_{kind}.py", controller_source(kind))
                   for kind in ("native", "model", "cpu_local8_standard", "cpu_local8_screen"))
    for name, source in outputs:
        with (ROOT / name).open("x") as stream:
            stream.write(source)
        print(ROOT / name)

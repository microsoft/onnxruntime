from pathlib import Path

from build_hc_owned_vector_unroll import NAME
from prepare_hc_owned_vector_proofs import (
    benchmark_command as original_command,
    benchmark_source as original_benchmark,
    controller_source as original_controller,
)


ROOT = Path(__file__).parent
BENCHMARK = ROOT / "benchmark_webgpu_int4_hc_owned_vector_unroll.py"


def adapt(source):
    return source.replace("hc_owned_vector", "hc_owned_vector_unroll").replace(
        "hc-owned-vector", "hc-owned-vector-unroll").replace(
        "ORT_WEBGPU_HC_OWNED_VECTOR", "ORT_WEBGPU_HC_OWNED_VECTOR_UNROLL").replace(
        "HCPostNormProjectionOwnedVectorF16", NAME)


def benchmark_source():
    return adapt(original_benchmark())


def benchmark_command(command):
    command = original_command(command)
    command[-1] = command[-1].replace("benchmark_webgpu_int4_hc_owned_vector.py", BENCHMARK.name)
    return command


def controller_source(kind):
    return adapt(original_controller(kind))


if __name__ == "__main__":
    outputs = [(BENCHMARK.name, benchmark_source())]
    outputs.extend((f"run_hc_owned_vector_unroll_{kind}.py", controller_source(kind))
                   for kind in ("native", "model", "cpu_local8_standard", "cpu_local8_screen"))
    outputs.append(("run_hc_owned_vector_unroll_invalid.py",
                    adapt((ROOT / "run_hc_owned_vector_invalid.py").read_text())))
    for name, source in outputs:
        with (ROOT / name).open("x") as stream:
            stream.write(source)
        print(ROOT / name)

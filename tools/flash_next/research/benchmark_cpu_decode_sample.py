import argparse
import functools
import json
import os
from pathlib import Path
import time

import benchmark_gpu_slice as benchmark


START_INDEX = 32
STOP_INDEX = 224


def instrument_requests(original, prefix):
    calls = 0

    @functools.wraps(original)
    def request(*args, **kwargs):
        nonlocal calls
        current = calls
        calls += 1
        if current == 1:
            if kwargs.get("on_decode_step") is not None:
                raise ValueError("A decode callback is already installed")

            def sample(index, timestamp):
                if index not in (START_INDEX, STOP_INDEX):
                    return
                phase = "start" if index == START_INDEX else "stop"
                marker = Path(str(prefix) + f".{phase}.json")
                record = dict(phase=phase, request_index=current, token_index=index,
                    container_pid=os.getpid(), monotonic_ns=time.monotonic_ns(),
                    token_timestamp=timestamp, sampling_scope="diagnostic_decode_only")
                temporary = Path(str(marker) + ".tmp")
                with temporary.open("x") as stream:
                    stream.write(json.dumps(record, indent=2) + "\n")
                os.link(temporary, marker)
                temporary.unlink()

            kwargs["on_decode_step"] = sample
        return original(*args, **kwargs)

    return request


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args()
    prefix = Path(os.environ["ORT_WEBGPU_CPU_PERF_MARKER_PREFIX"])
    flags = {key: os.environ.get(key) for key in ("ORT_WEBGPU_QMOE_FC2_PRIVATE_A",
        "ORT_GENAI_WEBGPU_GREEDY_ARGMAX_FUSED_MIN_LENGTH", "ORT_GENAI_WEBGPU_GPU_SLICE_COPY")}
    if flags != {"ORT_WEBGPU_QMOE_FC2_PRIVATE_A": "0",
                 "ORT_GENAI_WEBGPU_GREEDY_ARGMAX_FUSED_MIN_LENGTH": "0",
                 "ORT_GENAI_WEBGPU_GPU_SLICE_COPY": "1"}:
        raise ValueError("CPU diagnostic must use the supported ordinary-greedy/sliced-copy baseline")
    original = benchmark.run_request
    benchmark.run_request = instrument_requests(original, prefix)
    try:
        benchmark.main()
    finally:
        benchmark.run_request = original
    markers = [json.loads(Path(str(prefix) + f".{phase}.json").read_text()) for phase in ("start", "stop")]
    if any(os.environ.get(key) != value for key, value in flags.items()):
        raise RuntimeError("Private selectors changed during the CPU diagnostic")
    raw = json.loads(args.output.read_text())
    raw["headline_eligible"] = False
    raw["cpu_sampling"] = dict(enabled=True, diagnostic_only=True, natural_timing_qualified=False,
        request_index=1, start_index=START_INDEX, stop_index=STOP_INDEX, markers=markers, flags=flags)
    args.output.write_text(json.dumps(raw, indent=2) + "\n")

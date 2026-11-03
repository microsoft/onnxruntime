from functools import partial
import json
from pathlib import Path
import sys
import time

import benchmark_webgpu_int4 as benchmark


def profile_request(original, stream, og, model, tokenizer, prompt, output_tokens, chunk, *, maps_path=None, **options):
    if options:
        raise ValueError("CPU phase diagnostic supports only the unchanged standard request route")
    phase = dict(context=len(prompt), output_tokens=output_tokens, request_start=time.perf_counter())

    def boundary(index, timestamp):
        if index == 0:
            phase["decode_start"] = timestamp
            if maps_path is not None and not maps_path.exists():
                with maps_path.open("x") as mappings:
                    mappings.write(Path("/proc/self/maps").read_text())
        if index == output_tokens - 1:
            phase["decode_end"] = timestamp

    result = original(og, model, tokenizer, prompt, output_tokens, chunk, on_decode_step=boundary)
    assert phase["request_start"] < phase["decode_start"] < phase["decode_end"]
    assert abs(result["decode_seconds"] - (phase["decode_end"] - phase["decode_start"])) < 1e-8
    stream.write(json.dumps(phase) + "\n")
    stream.flush()
    return result


def main():
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    with output.with_suffix(".cpu-phases.jsonl").open("x") as stream:
        benchmark.run_request = partial(profile_request, benchmark.run_request, stream,
                                        maps_path=output.with_suffix(".cpu-maps.txt"))
        benchmark.main()


if __name__ == "__main__":
    main()

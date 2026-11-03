import argparse
import json
import os
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=Path, default=Path("/home/sayanshaw/repos/qwen38-webgpu-raw"))
source = parser.parse_args().model
paths = [source / "engram.data", source / "model.onnx.data"]
required = sum(path.stat().st_size for path in paths) + 32 * 1024**3
available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                 if line.startswith("MemAvailable:"))
if available < required:
    raise RuntimeError(f"Insufficient host memory for controlled read-only cache warming: {available} < {required}")
buffer = bytearray(8 * 1024**2)
for path in paths:
    start = time.monotonic()
    total = 0
    with path.open("rb", buffering=0) as stream:
        os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
        while count := stream.readinto(buffer):
            total += count
    if total != path.stat().st_size:
        raise RuntimeError(f"Unexpected byte count for {path}: {total}")
    print(json.dumps({"path": str(path), "bytes_read": total,
                      "elapsed_s": time.monotonic() - start, "host_available_before": available}), flush=True)

import argparse
import json
import os
from pathlib import Path

from benchmark_gpu_slice import main, run_request


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args()
    flag = "ORT_WEBGPU_QMOE_FC2_PRIVATE_A"
    selected = os.environ.get(flag)
    if selected not in ("0", "1"):
        raise ValueError(f"{flag} must explicitly select 0 or 1")
    metadata = Path(str(args.output) + ".launcher.json")
    with metadata.open("x") as stream:
        stream.write(json.dumps(dict(status="flags_captured_before_model_load",
                                     flags={flag: selected}), indent=2) + "\n")
    main()
    if os.environ.get(flag) != selected:
        raise RuntimeError("Process-static selector changed during benchmark")

import json
import math
from pathlib import Path

import numpy as np
import onnx

from build_qmoe_node_replay import sha
from prepare_dsa_subgroup import prepare_case


DIRECTORY = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill1280/fixtures-dsa-v1")


def main(directory=DIRECTORY, sequences=(768, 1280), boundary=1280):
    directory.mkdir(parents=True, exist_ok=False)
    positive = []
    for sequence in sequences:
        cases = (
            (f"selected{sequence}", dict(sequence=sequence, head_size=32)),
            (f"local_sink{sequence}", dict(sequence=sequence, heads=24, head_size=256, selected=33,
                local_window=128, auxiliary=64, shared_auxiliary=True, sink=True)),
            (f"aux_smooth{sequence}", dict(sequence=sequence, dtype=np.float32, head_size=32,
                auxiliary=19, shared_auxiliary=True, local_window=8, smooth=True)),
            (f"empty{sequence}", dict(sequence=sequence, selected=0, empty=True)),
        )
        for name, options in cases:
            prepare_case(directory, name, **options)
            positive.append(name)
    legacy = ["legacy256", "legacy512", "legacy1024"]
    for sequence, name in zip((256, 512, 1024), legacy):
        prepare_case(directory, name, sequence=sequence, head_size=32)
        positive.append(name)
    for name, options in (
        (f"reject{boundary-1}", dict(sequence=boundary-1)),
        (f"reject{boundary+1}", dict(sequence=boundary+1)),
        ("reject_batch2", dict(sequence=boundary, batch=2)),
        ("reject_dispatch", dict(sequence=boundary, heads=128)),
    ):
        options.update(kv_heads=1, head_size=8, selected=0, empty=True)
        if "heads" not in options:
            options["heads"] = 1
        prepare_case(directory, name, **options)
    manifest = dict(positive_cases=positive, legacy_cases=legacy, fixtures={}, original_model_unchanged=True)
    for path in sorted(directory.glob("*.onnx")):
        graph = onnx.load(path).graph
        elements = math.prod(d.dim_value for d in graph.output[0].type.tensor_type.shape.dim)
        manifest["fixtures"][path.name] = dict(sha256=sha(path), output_elements=elements)
    assert len(manifest["fixtures"]) == 2 * (len(positive) + 4)
    with (directory / "manifest.json").open("x") as stream:
        stream.write(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "fixtures_prepared_not_gpu_qualified", "files": len(manifest["fixtures"])}))


if __name__ == "__main__":
    main()

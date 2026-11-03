import json
import math
from pathlib import Path

import numpy as np
import onnx

from build_qmoe_node_replay import sha
from prepare_dsa_subgroup import prepare_case


DIRECTORY = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill512/fixtures-dsa-v4")
POSITIVE = ("selected512", "local_sink512", "aux_smooth512", "empty512", "legacy256")


def main():
    DIRECTORY.mkdir(parents=True, exist_ok=False)
    cases = (
        ("selected512", dict(sequence=512, head_size=32)),
        ("local_sink512", dict(sequence=512, heads=24, head_size=256, selected=33,
                               local_window=128, auxiliary=64, shared_auxiliary=True, sink=True)),
        ("aux_smooth512", dict(sequence=512, dtype=np.float32, head_size=32, auxiliary=19,
                               shared_auxiliary=True, local_window=8, smooth=True)),
        ("empty512", dict(sequence=512, selected=0, empty=True)),
        ("legacy256", dict(sequence=256, head_size=32)),
        ("reject511", dict(sequence=511, heads=1, kv_heads=1, head_size=8, selected=0, empty=True)),
        ("reject513", dict(sequence=513, heads=1, kv_heads=1, head_size=8, selected=0, empty=True)),
        ("reject_batch2", dict(sequence=512, batch=2, heads=1, kv_heads=1, head_size=8, selected=0, empty=True)),
        ("reject_dispatch", dict(sequence=512, heads=128, kv_heads=1, head_size=8, selected=0, empty=True)),
    )
    for name, options in cases:
        prepare_case(DIRECTORY, name, **options)
    manifest = dict(positive_cases=list(POSITIVE), fixtures={}, original_model_unchanged=True)
    for path in sorted(DIRECTORY.glob("*.onnx")):
        graph = onnx.load(path).graph
        elements = math.prod(d.dim_value for d in graph.output[0].type.tensor_type.shape.dim)
        manifest["fixtures"][path.name] = dict(sha256=sha(path), output_elements=elements)
    assert len(manifest["fixtures"]) == 2 * len(cases) == 18
    (DIRECTORY / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "fixtures_prepared_not_gpu_qualified", "files": len(manifest["fixtures"])}))


if __name__ == "__main__":
    main()

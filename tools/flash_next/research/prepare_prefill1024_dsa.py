import json
import math
from pathlib import Path

import numpy as np
import onnx

from build_qmoe_node_replay import sha
from prepare_dsa_subgroup import prepare_case


DIRECTORY = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill1024/fixtures-dsa-v1")


if __name__ == "__main__":
    DIRECTORY.mkdir(parents=True, exist_ok=False)
    cases = (
        ("selected1024", dict(sequence=1024, head_size=32)),
        ("local_sink1024", dict(sequence=1024, heads=24, head_size=256, selected=33,
                                local_window=128, auxiliary=64, shared_auxiliary=True, sink=True)),
        ("aux_smooth1024", dict(sequence=1024, dtype=np.float32, head_size=32, auxiliary=19,
                                shared_auxiliary=True, local_window=8, smooth=True)),
        ("empty1024", dict(sequence=1024, selected=0, empty=True)),
        ("legacy512", dict(sequence=512, head_size=32)),
        ("legacy256", dict(sequence=256, head_size=32)),
        ("reject1023", dict(sequence=1023, heads=1, kv_heads=1, head_size=8, selected=0, empty=True)),
        ("reject1025", dict(sequence=1025, heads=1, kv_heads=1, head_size=8, selected=0, empty=True)),
        ("reject_batch2", dict(sequence=1024, batch=2, heads=1, kv_heads=1, head_size=8, selected=0, empty=True)),
        ("reject_dispatch", dict(sequence=1024, heads=128, kv_heads=1, head_size=8, selected=0, empty=True)),
    )
    for name, options in cases:
        prepare_case(DIRECTORY, name, **options)
    manifest = dict(positive_cases=[name for name, _ in cases[:6]],
                    legacy_cases=["legacy256", "legacy512"], fixtures={}, original_model_unchanged=True)
    for path in sorted(DIRECTORY.glob("*.onnx")):
        graph = onnx.load(path).graph
        elements = math.prod(d.dim_value for d in graph.output[0].type.tensor_type.shape.dim)
        manifest["fixtures"][path.name] = dict(sha256=sha(path), output_elements=elements)
    assert len(manifest["fixtures"]) == 20
    (DIRECTORY / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "fixtures_prepared_not_gpu_qualified", "files": len(manifest["fixtures"])}))

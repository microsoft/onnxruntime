from pathlib import Path

from build_adapter_inline_tensors import FLAG, transform as original_transform
from build_adapter_tensor_reuse_sdk import main as build_sdk
from build_qmoe_node_replay import replace_once


OUTPUT = Path("/workspace/qwen38-int4-adapter-inline-tensors/build-v3")


def transform(source):
    source = original_transform(source)
    for old, new in (
        ("InlinedVector<Tensor, 8>", "InlinedVector<Tensor, 17>"),
        ('\\"input_capacity\\":8', '\\"input_capacity\\":17'),
        ("inputs > 8", "inputs > 17"),
    ):
        source = replace_once(source, old, new)
    return source


if __name__ == "__main__":
    build_sdk(OUTPUT, transform, FLAG, object_layout_changes=True)

from pathlib import Path

from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once
from build_tensor_cross_fma_probe import transform as cross_fma
from build_tensor_dot4_probe import FLAG


OUTPUT = Path("/workspace/qwen38-int4-tensor-dot4-reverse-cross-fma-probe/build-v1")


def transform(source):
    source = cross_fma(source, materialize=True)
    for original, replacement in (
        ('"TensorCrossFmaProbeF16"', '"TensorReverseCrossFmaProbeF16"'),
        ('a.GetByOffset("row")', 'a.GetByOffset("(row ^ 2u)")'),
        ('a.GetByOffset("col")', 'a.GetByOffset("(col ^ 2u)")'),
        ("((row % 8u) * 4u)", "(((row ^ 2u) % 8u) * 4u)"),
    ):
        source = replace_once(source, original, replacement)
    return source


if __name__ == "__main__":
    build(output=OUTPUT, flag=FLAG, source_transform=transform,
          component_dir=Path("/workspace/qwen38-int4-dense-block32/build-v1"),
          source_name="matmul_nbits.cc", comparator_flag="ORT_WEBGPU_DENSE_BLOCK32",
          comparator_transform=lambda source: source,
          parent_dir=Path("/workspace/qwen38-int4-qsa-prefill-score16/build-v1"),
          preserves_reduction=False)

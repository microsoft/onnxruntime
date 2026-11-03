from pathlib import Path

from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once
from build_tensor_dot4_probe import FLAG
from build_tensor_product_probe import transform as products


OUTPUT = Path("/workspace/qwen38-int4-tensor-dot4-pairwise-probe/build-v1")
PAIRWISE = """  let pair_bits = pack2x16float(vec2<f32>(f32(p.x) + f32(p.y),
                                               f32(p.z) + f32(p.w)));
  let pairs = bitcast<vec2<f16>>(pair_bits);
  return pairs.x + pairs.y;"""


def transform(source):
    source = products(source)
    source = replace_once(source, '"TensorProductProbeF16"', '"TensorProductPairwiseProbeF16"')
    return replace_once(source, "  return dot(vec4<f16>(1), p);", PAIRWISE)


if __name__ == "__main__":
    build(output=OUTPUT, flag=FLAG, source_transform=transform,
          component_dir=Path("/workspace/qwen38-int4-dense-block32/build-v1"),
          source_name="matmul_nbits.cc", comparator_flag="ORT_WEBGPU_DENSE_BLOCK32",
          comparator_transform=lambda source: source,
          parent_dir=Path("/workspace/qwen38-int4-qsa-prefill-score16/build-v1"),
          preserves_reduction=False)

from pathlib import Path

from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once
from build_tensor_dot4_probe import FLAG
from build_tensor_product_probe import transform as products


OUTPUT = Path("/workspace/qwen38-int4-tensor-dot4-f32-products-probe/build-v2")


def transform(source):
    source = products(source)
    for original, replacement in (
        ('"TensorProductProbeF16"', '"TensorProductF32ProbeF16"'),
        ("products: array<f16, 256>", "products: array<f32, 256>"),
        ("let p = vec4<f16>(products[", "let p = vec4<f32>(products["),
        ("return dot(vec4<f16>(1), p);", "return f16(dot(vec4<f32>(1), p));"),
        ("subgroup_matrix_result<f16, 16, 16>", "subgroup_matrix_result<f32, 16, 16>"),
        ("{{f16, f16, 16, 16, 16, 32, true}}",
         "{{f16, wgpu::SubgroupMatrixComponentType::F32, 16, 16, 16, 32, true}}"),
        ("Tensor dot4 probe requires native FP16 16x16x16 matrices",
         "Tensor product probe requires native FP16 inputs and FP32-result 16x16x16 matrices"),
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

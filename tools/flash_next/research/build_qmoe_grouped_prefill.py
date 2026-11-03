from pathlib import Path

from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent
FLAG = "ORT_WEBGPU_QMOE_GROUPED_PREFILL"
OUTPUT = Path("/workspace/qwen38-int4-qmoe-grouped-prefill/build-v1")
COMPONENT = Path("/workspace/qwen38-int4-qmoe-prefill1024/build-v1")
PARENT = Path("/workspace/qwen38-int4-qsa-prefill-score16/build-v1")


def transform(source):
    source = replace_once(source, "class BlockFp8ExpertMatMulProgram",
                          (ROOT / "qmoe_grouped_prefill.inc").read_text() + "\nclass BlockFp8ExpertMatMulProgram")
    validation = ('  ORT_RETURN_IF_NOT(prefill1024 == 0 || prefill1024 == 1, '
                  '"ORT_WEBGPU_QMOE_PREFILL1024 must be 0 or 1");')
    assert source.count(validation) == 2
    source = source.replace(validation, validation + f'''
  const int grouped_prefill = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(grouped_prefill == 0 || grouped_prefill == 1, "{FLAG} must be 0 or 1");''')
    source = replace_once(source, "  key.push_back(prefill1024);",
                          "  key.push_back(prefill1024);\n  key.push_back(grouped_prefill);")
    source = replace_once(source, "  Status status;\n",
        """  const bool use_grouped_prefill = grouped_prefill == 1 && use_extended_int4_prefill &&
      (use_prefill512 || use_prefill1024) && use_small_batch && !use_int4_cooperative;
  Status status;
""")
    source = replace_once(source, "    std::optional<Tensor> expanded_input;", """
    std::optional<Tensor> group_counts, group_offsets, group_rows, sorted_experts;
    if (use_grouped_prefill) {
      group_counts.emplace(context.CreateGPUTensor(dtype_uint32, TensorShape({num_experts})));
      group_offsets.emplace(context.CreateGPUTensor(dtype_uint32, TensorShape({num_experts + 1u})));
      group_rows.emplace(context.CreateGPUTensor(dtype_uint32, TensorShape({routed_rows})));
      sorted_experts.emplace(context.CreateGPUTensor(dtype_uint32, TensorShape({routed_rows})));
      QmoeGroupedCountProgram count;
      count.AddInput({&indirect_experts, ProgramTensorMetadataDependency::Type})
          .AddOutput({&*group_counts, ProgramTensorMetadataDependency::Type})
          .AddUniformVariables({{routed_rows}})
          .SetWorkgroupSize(64).SetDispatchGroupSize(num_experts);
      ORT_RETURN_IF_ERROR(context.RunProgram(count));
      QmoeGroupedOffsetProgram offsets;
      offsets.AddInput({&*group_counts, ProgramTensorMetadataDependency::Type})
          .AddOutput({&*group_offsets, ProgramTensorMetadataDependency::Type})
          .AddUniformVariables({{num_experts}})
          .SetWorkgroupSize(64).SetDispatchGroupSize(1);
      ORT_RETURN_IF_ERROR(context.RunProgram(offsets));
      QmoeGroupedPermutationProgram permutation;
      permutation.AddInputs({{&indirect_experts, ProgramTensorMetadataDependency::Type},
                             {&*group_offsets, ProgramTensorMetadataDependency::Type}})
          .AddOutputs({{&*group_rows, ProgramTensorMetadataDependency::Type},
                       {&*sorted_experts, ProgramTensorMetadataDependency::Type}})
          .AddUniformVariables({{routed_rows}})
          .SetWorkgroupSize(64).SetDispatchGroupSize(num_experts);
      ORT_RETURN_IF_ERROR(context.RunProgram(permutation));
    }
    const Tensor* projection_experts = use_grouped_prefill ? &*sorted_experts : &indirect_experts;
    std::optional<Tensor> expanded_input;""")
    original = """      ExpandMoEInputProgram expand;
      expand.AddInput({hidden_state, ProgramTensorMetadataDependency::Type})
          .AddOutput({&*expanded_input, ProgramTensorMetadataDependency::None})
          .SetWorkgroupSize(64)
          .SetDispatchGroupSize((hidden_size + 63) / 64, routed_rows)
          .AddUniformVariables({hidden_size, k, routed_rows});
      ORT_RETURN_IF_ERROR(context.RunProgram(expand));"""
    source = replace_once(source, original, """      if (use_grouped_prefill) {
        QmoeGroupedCopyProgram gather{false};
        gather.AddInputs({{hidden_state, ProgramTensorMetadataDependency::Type},
                          {&*group_rows, ProgramTensorMetadataDependency::Type}})
            .AddOutput({&*expanded_input, ProgramTensorMetadataDependency::Type})
            .AddUniformVariables({{routed_rows}, {hidden_size}, {k}})
            .SetWorkgroupSize(64).SetDispatchGroupSize((hidden_size + 63) / 64, routed_rows);
        ORT_RETURN_IF_ERROR(context.RunProgram(gather));
      } else {
""" + original + "\n      }")
    for output in ("fc1_outputs", "fc2_outputs"):
        source = replace_once(source, f"context, &{output}, 0, &indirect_experts,",
                              f"context, &{output}, 0, projection_experts,")
    source = replace_once(source, "    const uint32_t mix_wg_size = 256;", """
    std::optional<Tensor> restored_fc2;
    const Tensor* final_mix_input = &fc2_outputs;
    if (use_grouped_prefill) {
      restored_fc2.emplace(context.CreateGPUTensor(dtype, fc2_output_shape));
      QmoeGroupedCopyProgram scatter{true};
      scatter.AddInputs({{&fc2_outputs, ProgramTensorMetadataDependency::Type},
                         {&*group_rows, ProgramTensorMetadataDependency::Type}})
          .AddOutput({&*restored_fc2, ProgramTensorMetadataDependency::Type})
          .AddUniformVariables({{routed_rows}, {hidden_size}, {k}})
          .SetWorkgroupSize(64).SetDispatchGroupSize((hidden_size + 63) / 64, routed_rows);
      ORT_RETURN_IF_ERROR(context.RunProgram(scatter));
      final_mix_input = &*restored_fc2;
    }
    const uint32_t mix_wg_size = 256;""")
    return replace_once(source,
                        "    final_mix\n        .AddInputs({{&fc2_outputs, ProgramTensorMetadataDependency::Type}})",
                        "    final_mix\n        .AddInputs({{final_mix_input, ProgramTensorMetadataDependency::Type}})")


if __name__ == "__main__":
    build(output=OUTPUT, flag=FLAG, source_transform=transform, component_dir=COMPONENT,
          source_name="qmoe.cc", comparator_flag="ORT_WEBGPU_QMOE_PREFILL1024",
          comparator_transform=lambda source: replace_once(source, "prefill1024-v1-native-",
                                                           "grouped-prefill-v1-native-"),
          parent_dir=PARENT,
          extra_include_dirs=(Path("/workspace/qwen38-int4-qmoe-prefill256/build-v1/overlay/"
                                   "contrib_ops/webgpu/moe"),))

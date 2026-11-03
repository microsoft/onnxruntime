from pathlib import Path

from build_hc_bitcast_unpack import main as build
from build_qmoe_grouped_prefill import FLAG
from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent
PROGRAM = "MatMulNBitsGroupedRows4F16"
GROUPED = Path("/workspace/qwen38-int4-qmoe-grouped-prefill/build-v1")
OUTPUT = Path("/workspace/qwen38-int4-qmoe-prefill-rows4/build-v2")


def transform(source):
    source = replace_once(source, "class MatMulInt3Program final",
                          (ROOT / "qmoe_prefill_rows4.inc").read_text() + "\nclass MatMulInt3Program final")
    start = source.index("Status ApplyMatMulNBits(")
    anchor = "  if (nbits == 3) {"
    position = source.index(anchor, start)
    branch = f'''
  const int grouped_prefill_rows = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(grouped_prefill_rows == 0 || grouped_prefill_rows == 1,
                    "{FLAG} must be 0 or 1");
  if (grouped_prefill_rows == 1 && batch_count == 1 && dispatch_M == M &&
      (M == 5120 || M == 10240) && ((K == 2560 && N == 1280) || (K == 640 && N == 2560)) &&
      nbits == 4 && block_size == 32 && has_weight_idx_indirect && !has_zero_points &&
      a->IsDataType<MLFloat16>() && scales->IsDataType<MLFloat16>() &&
      (bias == nullptr || bias->IsDataType<MLFloat16>()) &&
      !context.EnableMatmulFp32Accumulation() &&
      context.AdapterInfo().vendor == std::string_view{{"nvidia"}}) {{
    GroupedPrefillRows4Program program{{has_bias}};
    program.SetWorkgroupSize(128).SetDispatchGroupSize(N / 8, M / 4);
    program.AddInputs({{{{a, ProgramTensorMetadataDependency::TypeAndRank, 4}},
                        {{b, ProgramTensorMetadataDependency::TypeAndRank, 16}},
                        {{scales, ProgramTensorMetadataDependency::TypeAndRank, 1}}}});
    if (has_bias) program.AddInput({{bias, ProgramTensorMetadataDependency::TypeAndRank, 1}});
    program.AddInput({{weight_index_indirect, ProgramTensorMetadataDependency::TypeAndRank}})
        .AddOutput({{y, ProgramTensorMetadataDependency::TypeAndRank, 1}})
        .AddUniformVariables({{{{N}}, {{K}}, {{N / 8}}}})
        .CacheHint(has_bias);
    return context.RunProgram(program);
  }}

'''
    return source[:position] + branch + source[position:]


if __name__ == "__main__":
    build(output=OUTPUT, flag=FLAG, source_transform=transform,
          component_dir=Path("/workspace/qwen38-int4-dense-block32/build-v1"),
          source_name="matmul_nbits.cc", comparator_flag=FLAG, comparator_dir=GROUPED,
          comparator_transform=lambda source: replace_once(
              source, "grouped-prefill-v1-native-", "grouped-prefill-rows4-v2-native-"),
          parent_dir=GROUPED)

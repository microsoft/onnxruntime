from pathlib import Path

from build_qmoe_node_replay import replace_once
from build_qsa_concat_vector import main as build


FLAG = "ORT_WEBGPU_QSA_HIST_RANK"
PROGRAM = "QsaHistogramBoundedRank"
HISTOGRAM = "QsaRankHistogram"
UNIFORM_PROGRAM = "QsaHistogramUniformRank"
PARENT = Path("/workspace/qwen38-int4-qsa-prefill-score16/build-v1")
OUTPUT = Path("/workspace/qwen38-int4-qsa-hist-rank/build-v1")
BUCKET = """
fn qsa_score_bucket(score: f32, maximum_bits: u32) -> u32 {
  let maximum = bitcast<f32>(maximum_bits);
  if (maximum_bits == 0u || maximum_bits >= 0x7f800000u || maximum == 0.0) { return 0u; }
  return u32(clamp((score / maximum) * 255.0, 0.0, 255.0));
}
"""


def histogram_program():
    return r'''
class QsaRankHistogramProgram final : public Program<QsaRankHistogramProgram> {
 public:
  QsaRankHistogramProgram() : Program{"QsaRankHistogram"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"total_sequence_length", ProgramUniformVariableDataType::Uint32},
      {"compress_ratio", ProgramUniformVariableDataType::Uint32},
      {"block_topk", ProgramUniformVariableDataType::Uint32});

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& visible = shader.AddInput("visible", ShaderUsage::UseUniform);
    const auto& scores = shader.AddInput("scores", ShaderUsage::UseUniform);
    const auto& bounds = shader.AddOutput("bounds", ShaderUsage::UseUniform);
    shader.AdditionalImplementation() << R"WGSL(
var<workgroup> maxima: array<u32, 256>;
var<workgroup> bins: array<atomic<u32>, 256>;
''' + BUCKET + r''')WGSL";
    shader.MainFunctionBody()
        << "let count = " << visible.GetByOffset("uniforms.total_sequence_length") << " / uniforms.compress_ratio;\n"
        << "var biggest = 0u;\n"
        << "for (var other = local_idx; other < count; other += 256u) {\n"
        << "  let score = " << scores.GetByOffset("other") << ";\n"
        << "  if (score == score) {\n"
        << "    if (score < 0.0) { biggest = 0x7f800000u; }\n"
        << "    if (score > 0.0) { biggest = max(biggest, bitcast<u32>(score)); }\n"
        << "  }\n"
        << "}\n"
        << "maxima[local_idx] = biggest;\n"
        << "atomicStore(&bins[local_idx], 0u);\n"
        << "workgroupBarrier();\n"
        << "for (var width = 128u; width > 0u; width >>= 1u) {\n"
        << "  if (local_idx < width) { maxima[local_idx] = max(maxima[local_idx], maxima[local_idx + width]); }\n"
        << "  workgroupBarrier();\n"
        << "}\n"
        << "for (var other = local_idx; other < count; other += 256u) {\n"
        << "  let score = " << scores.GetByOffset("other") << ";\n"
        << "  if (score == score) { atomicAdd(&bins[qsa_score_bucket(score, maxima[0])], 1u); }\n"
        << "}\n"
        << "workgroupBarrier();\n"
        << "if (local_idx == 0u) {\n"
        << "  let selected = min(uniforms.block_topk, count);\n"
        << "  var total = 0u;\n"
        << "  var cutoff = 0u;\n"
        << "  for (var bucket = 256u; bucket > 0u; bucket--) {\n"
        << "    total += atomicLoad(&bins[bucket - 1u]);\n"
        << "    if (total >= selected) { cutoff = bucket - 1u; break; }\n"
        << "  }\n"
        << bounds.SetByOffset("0u", "maxima[0]") << "\n"
        << bounds.SetByOffset("1u", "cutoff") << "\n"
        << "}\n";
    return Status::OK();
  }
};
'''


def bounded_program(source, uniform_exit=False):
    start = source.index("class SparseAttentionIndexerQsaParallelRankProgram final")
    stop = source.index("\nStatus CheckShape(", start)
    program = source[start:stop].replace("SparseAttentionIndexerQsaParallelRankProgram",
                                        "QsaHistogramBoundedRankProgram")
    program = replace_once(program, 'Program{"SparseAttentionIndexerQsaParallelRank"}',
                           f'Program{{"{PROGRAM}"}}')
    marker = '    const auto& scores = shader.AddInput("scores", ShaderUsage::UseUniform);'
    program = replace_once(program, marker, marker +
                           '\n    const auto& bounds = shader.AddInput("bounds", ShaderUsage::UseUniform);')
    marker = '<< "var<workgroup> valid_counts: array<u32, 64>;\\n";'
    program = replace_once(program, marker, marker[:-1] + '\n        << R"WGSL(' + BUCKET + ')WGSL";')
    marker = '        << "  var rank = 0u;\\n"'
    program = replace_once(program, marker,
        '''        << "  let pruned = candidate != 0u && candidate < block_count && score == score && "
        << "qsa_score_bucket(score, " << bounds.GetByOffset("0u") << ") < " << bounds.GetByOffset("1u") << ";\\n"
''' + marker)
    program = replace_once(program,
        "if (candidate < block_count && first_score == first_score && score == score)",
        "if (!pruned && candidate < block_count && first_score == first_score && score == score)")
    # Pruned workgroups still reach every original barrier before their final writer returns.
    program = replace_once(program, '        << "  if (local_idx == 0u) {\\n"',
                        '        << "  if (local_idx == 0u) {\\n"\n'
                        '        << "    if (pruned) { return; }\\n"')
    if uniform_exit:
        program = program.replace("QsaHistogramBoundedRankProgram", "QsaHistogramUniformRankProgram")
        program = replace_once(program, f'Program{{"{PROGRAM}"}}', f'Program{{"{UNIFORM_PROGRAM}"}}')
        program = replace_once(program, BUCKET, "var<workgroup> pruning: u32;\n" + BUCKET)
        program = replace_once(program, "let pruned = candidate", "let should_prune = candidate")
        program = replace_once(program, '        << "  var rank = 0u;\\n"',
            '''        << "  if (local_idx == 0u) { pruning = select(0u, 1u, should_prune); }\\n"
        << "  let pruned = workgroupUniformLoad(&pruning) == 1u;\\n"
        << "  if (pruned) { return; }\\n"
        << "  var rank = 0u;\\n"''')
        program = replace_once(program, '        << "    if (pruned) { return; }\\n"\n', "")
    return program


def transform(source, minimum_stride=2048, uniform_exit=False):
    source = replace_once(source, "\nStatus CheckShape(",
                          "\n" + histogram_program() + bounded_program(source, uniform_exit) + "\nStatus CheckShape(")
    signature = "Status SparseAttentionIndexer::ComputeQsa(onnxruntime::webgpu::ComputeContext& context) const {"
    source = replace_once(source, signature, signature + f'''
  const int histogram_rank = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(histogram_rank == 0 || histogram_rank == 1, "{FLAG} must be 0 or 1");
''')
    start = source.index("    if (use_parallel_rank && phase == QsaCachePhase::Select) {")
    stop = source.index("    const uint32_t phase_sort_capacity", start)
    dispatch = source[start:stop]
    dispatch = replace_once(dispatch, "if (use_parallel_rank && phase == QsaCachePhase::Select)",
        """if (histogram_rank == 1 && use_parallel_rank && phase == QsaCachePhase::Select &&
        score_stride >= 2048 && context.DeviceLimits().maxComputeWorkgroupSizeX >= 256 &&
        context.DeviceLimits().maxComputeInvocationsPerWorkgroup >= 256 &&
        context.DeviceLimits().maxComputeWorkgroupStorageSize >= 2048)""")
    dispatch = replace_once(dispatch, "      SparseAttentionIndexerQsaParallelRankProgram rank;", """
      Tensor bounds = context.CreateGPUTensor(DataTypeImpl::GetType<uint32_t>(), TensorShape({2}));
      QsaRankHistogramProgram histogram;
      histogram.AddInputs({{&visible_cache, ProgramTensorMetadataDependency::Type},
                           {&score_cache, ProgramTensorMetadataDependency::Type}})
          .AddOutput({&bounds, ProgramTensorMetadataDependency::Type})
          .AddUniformVariables({{ToUint32(total_length)}, {ToUint32(compress_ratio_)}, {ToUint32(block_topk)}})
          .SetWorkgroupSize(256).SetDispatchGroupSize(1);
      ORT_RETURN_IF_ERROR(context.RunProgram(histogram));
      QsaHistogramBoundedRankProgram rank;""")
    dispatch = replace_once(dispatch,
        "rank.AddInputs({{&visible_cache, ProgramTensorMetadataDependency::Type},\n"
        "                      {&score_cache, ProgramTensorMetadataDependency::Type}})",
        "rank.AddInputs({{&visible_cache, ProgramTensorMetadataDependency::Type},\n"
        "                      {&score_cache, ProgramTensorMetadataDependency::Type},\n"
        "                      {&bounds, ProgramTensorMetadataDependency::Type}})")
    dispatch = replace_once(dispatch, "score_stride >= 2048", f"score_stride >= {minimum_stride}")
    if uniform_exit:
        dispatch = replace_once(dispatch, "QsaHistogramBoundedRankProgram rank;",
                                "QsaHistogramUniformRankProgram rank;")
    return source[:start] + dispatch + source[start:]


if __name__ == "__main__":
    build(output=OUTPUT, source_transform=transform, parent_path=PARENT, component_path=PARENT,
          no_arithmetic_changes=False)

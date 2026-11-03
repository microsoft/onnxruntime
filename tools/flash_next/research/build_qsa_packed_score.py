from pathlib import Path

from build_qmoe_node_replay import replace_once
from build_qsa_concat_vector import main as build_leaf


OUTPUT = Path("/workspace/qwen38-int4-qsa-packed-score/build-v2")
FLAG = "ORT_WEBGPU_QSA_PACKED_SCORE"


def transform(source):
    start = source.index("Status SparseAttentionIndexerQsaSelectProgram::GenerateShaderCode")
    stop = source.index("\nStatus ", start + 1)
    function = source[start:stop]
    score_start = function.index("    if (phase_ == QsaCachePhase::Score) {")
    score_stop = function.index("    if (phase_ == QsaCachePhase::Map) {", score_start)
    original_score = function[score_start:score_stop]
    score = original_score.replace("local_idx", "local_lane")
    score = replace_once(score, '"var<workgroup> pooled_key: array<f32, " << kQsaSharedHeadSize',
                         '"var<workgroup> pooled_key: array<f32, " << 4 * kQsaSharedHeadSize')
    score = replace_once(score, "head_scores: array<f32, 64>", "head_scores: array<f32, 256>")
    score = replace_once(score, "key_inverse_rms: f32", "key_inverse_rms: array<f32, 4>")
    score = score.replace("pooled_key[d]", "pooled_key[group * 128u + d]")
    score = score.replace("pooled_key[d + half]", "pooled_key[group * 128u + d + half]")
    score = score.replace("key_inverse_rms =", "key_inverse_rms[group] =")
    score = score.replace(" * key_inverse_rms * ", " * key_inverse_rms[group] * ")
    score = score.replace("head_scores[local_lane]", "head_scores[group * 64u + local_lane]")
    score = score.replace("head_scores[head]", "head_scores[group * 64u + head]")
    score = score.replace("d += 64u", "d += 16u")
    score = replace_once(score,
        '"  let row = workgroup_idx / stride;\\n"\n'
        '          << "  let candidate = workgroup_idx % stride;\\n"',
        '"  let group = local_idx / 16u;\\n"\n'
        '          << "  let local_lane = local_idx % 16u;\\n"\n'
        '          << "  let packed_stride = (stride + 3u) / 4u;\\n"\n'
        '          << "  let row = workgroup_idx / packed_stride;\\n"\n'
        '          << "  let candidate = (workgroup_idx % packed_stride) * 4u + group;\\n"')
    score = replace_once(score, "if (local_lane == 0u) { visible_count_shared",
                         "if (local_idx == 0u) { visible_count_shared")
    function = function[:score_start] + score + function[score_stop:]
    function = function.replace("SparseAttentionIndexerQsaSelectProgram", "QsaPackedScoreProgram")
    declaration = """class QsaPackedScoreProgram final : public Program<QsaPackedScoreProgram> {
 public:
  explicit QsaPackedScoreProgram(bool has_mask)
      : Program{"QsaPackedScore4"}, has_mask_{has_mask} {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"rows", ProgramUniformVariableDataType::Uint32},
      {"sequence_length", ProgramUniformVariableDataType::Uint32},
      {"num_heads", ProgramUniformVariableDataType::Uint32},
      {"head_size", ProgramUniformVariableDataType::Uint32},
      {"query_row_stride", ProgramUniformVariableDataType::Uint32},
      {"rotary_width", ProgramUniformVariableDataType::Uint32},
      {"max_rotary_length", ProgramUniformVariableDataType::Uint32},
      {"rotary_cache_batch_stride", ProgramUniformVariableDataType::Uint32},
      {"compress_ratio", ProgramUniformVariableDataType::Uint32},
      {"capacity", ProgramUniformVariableDataType::Uint32},
      {"past_sequence_length", ProgramUniformVariableDataType::Uint32},
      {"total_sequence_length", ProgramUniformVariableDataType::Uint32},
      {"block_topk", ProgramUniformVariableDataType::Uint32},
      {"epsilon", ProgramUniformVariableDataType::Float32},
      {"scale", ProgramUniformVariableDataType::Float32});
 private:
  bool has_mask_;
  const bool use_global_cache_ = true;
  const QsaCachePhase phase_ = QsaCachePhase::Score;
  const uint32_t sort_capacity_ = 0;
  const bool dense_map_ = false;
};

"""
    source = source[:start] + declaration + function + source[start:]
    start = source.index("    SparseAttentionIndexerQsaSelectProgram select{",
                         source.index("Status SparseAttentionIndexer::ComputeQsa"))
    stop = source.index("    ORT_RETURN_IF_ERROR(context.RunProgram(select));", start)
    stop += len("    ORT_RETURN_IF_ERROR(context.RunProgram(select));\n")
    dispatch = source[start:stop]
    dispatch = replace_once(dispatch,
        "SparseAttentionIndexerQsaSelectProgram select{\n"
        "        mask != nullptr, use_global_cache, phase, phase_sort_capacity, phase_dense_map};",
        "QsaPackedScoreProgram select{mask != nullptr};")
    dispatch = replace_once(dispatch,
        "ToUint32(use_global_cache && phase == QsaCachePhase::Score ? score_elements : rows)",
        "DivideRoundUp(ToUint32(score_stride), 4u)")
    anchor = "    if (use_parallel_rank && phase == QsaCachePhase::Select) {\n"
    packed_dispatch = f"""    if (packed_score == 1 && use_global_cache && rows == 1 &&
        num_heads <= 16 && score_stride >= 2048 &&
        context.AdapterInfo().vendor == std::string_view{{"nvidia"}} &&
        phase == QsaCachePhase::Score) {{
      const uint32_t phase_sort_capacity = 0;
      const bool phase_dense_map = false;
{dispatch}      continue;
    }}
"""
    source = replace_once(source, anchor, packed_dispatch + anchor)
    anchor = "  // Bound all-pairs ranking while exposing each candidate to an independent workgroup.\n"
    selector = f"""  const int packed_score = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(packed_score == 0 || packed_score == 1, "{FLAG} must be 0 or 1");
"""
    return replace_once(source, anchor, selector + anchor)


if __name__ == "__main__":
    build_leaf(output=OUTPUT, source_transform=transform)

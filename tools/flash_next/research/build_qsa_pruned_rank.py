from pathlib import Path

from build_qmoe_node_replay import replace_once
from build_qsa_concat_vector import main as build


FLAG = "ORT_WEBGPU_QSA_PRUNED_RANK"
COMPONENT = Path("/workspace/qwen38-int4-qsa-score16/build-v5")
PARENT = Path("/workspace/qwen38-int4-dense-block32/build-v2")
OUTPUT = Path("/workspace/qwen38-int4-qsa-pruned-rank/build-v1")


def pruned_program(source):
    start = source.index("class SparseAttentionIndexerQsaParallelRankProgram final")
    stop = source.index("\nStatus CheckShape(", start)
    program = source[start:stop].replace("SparseAttentionIndexerQsaParallelRankProgram", "QsaPrunedRankProgram")
    program = program.replace('Program{"SparseAttentionIndexerQsaParallelRank"}', 'Program{"QsaPrunedRank"}')
    start = program.index("    shader.AdditionalImplementation()")
    stop = program.index("    shader.MainFunctionBody()", start)
    program = program[:start] + program[stop:]
    start = program.index('        << "  var rank = 0u;')
    stop = program.index('        << "  if (local_idx == 0u)', start)
    program = program[:start] + '''        << "  var rank = 0u;\\n"
        << "  var valid_count = 0u;\\n"
        << "  if (candidate < block_count && first_score == first_score && score == score) {\\n"
        << "    for (var base = 0u; base < block_count; base += 32u) {\\n"
        << "      let other = base + local_idx;\\n"
        << "      if (other < block_count) {\\n"
        << "        let other_score = " << scores.GetByOffset("other") << ";\\n"
        << "        if (candidate == 0u && other_score == other_score) { valid_count++; }\\n"
        << "        if (other_score > score || (other_score == score && other < candidate)) { rank++; }\\n"
        << "      }\\n"
        << "      if (candidate != 0u && subgroupAdd(rank) >= selected_blocks) { return; }\\n"
        << "    }\\n"
        << "  }\\n"
        << "  let final_rank = subgroupAdd(rank);\\n"
        << "  let final_valid_count = subgroupAdd(valid_count);\\n"
''' + program[stop:]
    return program.replace("valid_counts[0]", "final_valid_count").replace("ranks[0]", "final_rank")


def transform(source):
    source = replace_once(source, "\nStatus CheckShape(", "\n" + pruned_program(source) + "\nStatus CheckShape(")
    signature = "Status SparseAttentionIndexer::ComputeQsa(onnxruntime::webgpu::ComputeContext& context) const {"
    source = replace_once(source, signature, signature + f'''
  const int pruned_rank_mode = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(pruned_rank_mode == 0 || pruned_rank_mode == 1, "{FLAG} must be 0 or 1");
''')
    start = source.index("    if (use_parallel_rank && phase == QsaCachePhase::Select) {")
    stop = source.index("    const uint32_t phase_sort_capacity", start)
    original = source[start:stop]
    dispatch = replace_once(original, "if (use_parallel_rank && phase == QsaCachePhase::Select)",
        '''if (pruned_rank_mode == 1 && use_parallel_rank && phase == QsaCachePhase::Select &&
        context.HasFeature(wgpu::FeatureName::Subgroups) &&
        context.AdapterInfo().subgroupMinSize == 32 && context.AdapterInfo().subgroupMaxSize == 32)''')
    dispatch = dispatch.replace("SparseAttentionIndexerQsaParallelRankProgram", "QsaPrunedRankProgram")
    dispatch = replace_once(dispatch, ".SetWorkgroupSize(kQsaSelectWorkgroupSize)", ".SetWorkgroupSize(32)")
    return source[:start] + dispatch + source[start:]


if __name__ == "__main__":
    build(output=OUTPUT, source_transform=transform, parent_path=PARENT, component_path=COMPONENT)

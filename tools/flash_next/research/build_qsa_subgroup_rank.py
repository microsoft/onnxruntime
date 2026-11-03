from pathlib import Path

from build_qmoe_node_replay import replace_once
from build_qsa_concat_vector import main as build
from build_qsa_pruned_rank import COMPONENT, PARENT


FLAG = "ORT_WEBGPU_QSA_SUBGROUP_RANK"
OUTPUT = Path("/workspace/qwen38-int4-qsa-subgroup-rank/build-v1")


def subgroup_program(source):
    start = source.index("class SparseAttentionIndexerQsaParallelRankProgram final")
    stop = source.index("\nStatus CheckShape(", start)
    program = source[start:stop].replace("SparseAttentionIndexerQsaParallelRankProgram", "QsaSubgroupRankProgram")
    program = program.replace('Program{"SparseAttentionIndexerQsaParallelRank"}', 'Program{"QsaSubgroupRank"}')
    program = program.replace("ranks: array<u32, 64>", "ranks: array<u32, 2>")
    program = program.replace("valid_counts: array<u32, 64>", "valid_counts: array<u32, 2>")
    start = program.index('        << "  ranks[local_idx] = rank;')
    stop = program.index('        << "  if (local_idx == 0u)', start)
    program = program[:start] + '''        << "  let subgroup_rank = subgroupAdd(rank);\\n"
        << "  let subgroup_valid_count = subgroupAdd(valid_count);\\n"
        << "  if (sg_id == 0u) {\\n"
        << "    ranks[subgroup_index] = subgroup_rank;\\n"
        << "    valid_counts[subgroup_index] = subgroup_valid_count;\\n"
        << "  }\\n"
        << "  workgroupBarrier();\\n"
''' + program[stop:]
    return program.replace("valid_counts[0]", "(valid_counts[0] + valid_counts[1])").replace(
        "rank = ranks[0];", "rank = ranks[0] + ranks[1];")


def transform(source):
    source = replace_once(source, "\nStatus CheckShape(", "\n" + subgroup_program(source) + "\nStatus CheckShape(")
    signature = "Status SparseAttentionIndexer::ComputeQsa(onnxruntime::webgpu::ComputeContext& context) const {"
    source = replace_once(source, signature, signature + f'''
  const int subgroup_rank_mode = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(subgroup_rank_mode == 0 || subgroup_rank_mode == 1, "{FLAG} must be 0 or 1");
''')
    start = source.index("    if (use_parallel_rank && phase == QsaCachePhase::Select) {")
    stop = source.index("    const uint32_t phase_sort_capacity", start)
    original = source[start:stop]
    dispatch = replace_once(original, "if (use_parallel_rank && phase == QsaCachePhase::Select)",
        '''if (subgroup_rank_mode == 1 && use_parallel_rank && phase == QsaCachePhase::Select &&
        context.HasFeature(wgpu::FeatureName::Subgroups) &&
        context.AdapterInfo().subgroupMinSize == 32 && context.AdapterInfo().subgroupMaxSize == 32)''')
    dispatch = dispatch.replace("SparseAttentionIndexerQsaParallelRankProgram", "QsaSubgroupRankProgram")
    return source[:start] + dispatch + source[start:]


if __name__ == "__main__":
    build(output=OUTPUT, source_transform=transform, parent_path=PARENT, component_path=COMPONENT)

from pathlib import Path

from build_qmoe_node_replay import replace_once
from build_qsa_concat_vector import main as build
from build_qsa_pruned_rank import COMPONENT, PARENT, transform as original_transform


OUTPUT = Path("/workspace/qwen38-int4-qsa-pruned-rank/build-v2")


def transform(source):
    source = original_transform(source)
    start = source.index("class QsaPrunedRankProgram final")
    stop = source.index("\nStatus CheckShape(", start)
    program = source[start:stop]
    start_body = program.index("    shader.MainFunctionBody()")
    start_rank = program.index('        << "  var rank = 0u;', start_body)
    program = program[:start_body] + '''    shader.AdditionalImplementation()
        << "var<workgroup> rank_header: vec4<u32>;\\n"
        << "var<workgroup> rank_rejected: u32;\\n";
    shader.MainFunctionBody()
        << "  let candidate = workgroup_idx;\\n"
        << "  if (candidate >= uniforms.score_stride) { return; }\\n"
        << "  if (local_idx == 0u) {\\n"
        << "    let count = " << visible.GetByOffset("uniforms.total_sequence_length") << ";\\n"
        << "    var first = 0.0;\\n"
        << "    var value = 0.0;\\n"
        << "    if (count / uniforms.compress_ratio > 0u) { first = " << scores.GetByOffset("0u") << "; }\\n"
        << "    if (candidate < count / uniforms.compress_ratio) { value = " << scores.GetByOffset("candidate") << "; }\\n"
        << "    rank_header = vec4<u32>(count, bitcast<u32>(first), bitcast<u32>(value), 0u);\\n"
        << "  }\\n"
        << "  let header = workgroupUniformLoad(&rank_header);\\n"
        << "  let visible_count = header.x;\\n"
        << "  let first_score = bitcast<f32>(header.y);\\n"
        << "  let score = bitcast<f32>(header.z);\\n"
        << "  let block_count = visible_count / uniforms.compress_ratio;\\n"
        << "  let selected_blocks = min(uniforms.block_topk, block_count);\\n"
''' + program[start_rank:]
    program = replace_once(program,
        '''        << "    for (var base = 0u; base < block_count; base += 32u) {\\n"
        << "      let other = base + local_idx;\\n"''',
        '''        << "    for (var base = 0u; base < block_count; base += 512u) {\\n"
        << "      for (var offset = local_idx; offset < 512u; offset += 32u) {\\n"
        << "      let other = base + offset;\\n"''')
    program = replace_once(program,
        '''        << "      if (candidate != 0u && subgroupAdd(rank) >= selected_blocks) { return; }\\n"''',
        '''        << "      }\\n"
        << "      let current_rank = subgroupAdd(rank);\\n"
        << "      if (local_idx == 0u) {\\n"
        << "        rank_rejected = select(0u, 1u, candidate != 0u && current_rank >= selected_blocks);\\n"
        << "      }\\n"
        << "      if (workgroupUniformLoad(&rank_rejected) != 0u) { return; }\\n"''')
    return source[:start] + program + source[stop:]


if __name__ == "__main__":
    build(output=OUTPUT, source_transform=transform, parent_path=PARENT, component_path=COMPONENT)

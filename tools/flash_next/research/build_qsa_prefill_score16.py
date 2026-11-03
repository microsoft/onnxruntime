from pathlib import Path

from build_qmoe_node_replay import replace_once
from build_qsa_concat_vector import main as build


FLAG = "ORT_WEBGPU_QSA_PREFILL_SCORE16"
COMPONENT = Path("/workspace/qwen38-int4-qsa-score16/build-v5")
PARENT = Path("/workspace/qwen38-int4-dense-block32/build-v2")
OUTPUT = Path("/workspace/qwen38-int4-qsa-prefill-score16/build-v1")
ANCHOR = "    if (score16 == 1 && (packed_score == 1 || indexer_stack == 1) && use_global_cache && rows == 1 &&"
STOP = "    if ((packed_score == 1 || indexer_stack == 1) && use_global_cache && rows == 1 &&"


def transform(source):
    signature = "Status SparseAttentionIndexer::ComputeQsa(onnxruntime::webgpu::ComputeContext& context) const {"
    source = replace_once(source, signature, signature + f'''
  const int prefill_score16 = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(prefill_score16 == 0 || prefill_score16 == 1, "{FLAG} must be 0 or 1");
''')
    start, stop = source.index(ANCHOR), source.index(STOP)
    original = source[start:stop]
    dispatch = replace_once(original, ANCHOR, '''    if (prefill_score16 == 1 && indexer_stack == 1 && use_global_cache &&
        batch_size == 1 && (rows == 512 || rows == 1024) && rows == sequence_length &&
        query->IsDataType<MLFloat16>() && present->IsDataType<MLFloat16>() &&
        num_heads == 4 && head_size == 128 && query_shape[2] == 640 &&
        compress_ratio_ == 4 && token_budget_ == 2048 &&
        rows <= context.DeviceLimits().maxComputeWorkgroupsPerDimension &&
        DivideRoundUp(ToUint32(score_stride), 16u) <= context.DeviceLimits().maxComputeWorkgroupsPerDimension &&''')
    dispatch = replace_once(dispatch, "num_heads <= 4 && score_stride >= 2048",
                            "score_stride >= 128 && score_stride <= 16384")
    dispatch = replace_once(dispatch, "phase_sort_capacity, phase_dense_map)",
                            'phase_sort_capacity, phase_dense_map, "prefill-score16-2d")')
    dispatch = replace_once(dispatch, ".SetDispatchGroupSize(DivideRoundUp(ToUint32(score_stride), 16u))",
                            ".SetDispatchGroupSize(DivideRoundUp(ToUint32(score_stride), 16u), ToUint32(rows))")
    return source[:start] + dispatch + source[start:]


if __name__ == "__main__":
    build(output=OUTPUT, source_transform=transform, parent_path=PARENT, component_path=COMPONENT)

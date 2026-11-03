import argparse
from pathlib import Path

from build_qmoe_node_replay import replace_once
from build_qsa_concat_vector import main as build_leaf


PARENT = Path("/workspace/qwen38-int4-qmoe-prefill256/build-v1")
COMPONENT = Path("/workspace/qwen38-int4-qsa-indexer-stack/build-v1")
OUTPUT = Path("/workspace/qwen38-int4-qsa-score16/build-v2")
FLAG = "ORT_WEBGPU_QSA_SCORE16"
STORAGE_BYTES = 16 * 128 * 4 + 16 * 4 * 4 + 16 * 4 + 4
DISPATCH_ANCHOR = "    if ((packed_score == 1 || indexer_stack == 1) && use_global_cache && rows == 1 &&\n"


def transform(source, minimum_stride=8192, pad_keys=False, cache_query=False, wide_group=False):
    assert not wide_group or (pad_keys and cache_query)
    storage_bytes = STORAGE_BYTES + (16 * (16 if wide_group else 4) * 4 if pad_keys else 0)
    if cache_query:
        storage_bytes += 4 * 129 * 4 + 4 * 4
    start = source.index("class QsaPackedScoreProgram final")
    function = source.index("Status QsaPackedScoreProgram::GenerateShaderCode", start)
    stop = source.index("\nStatus ", function + 1)
    private = source[start:stop]
    score_start = private.index("    if (phase_ == QsaCachePhase::Score) {")
    score_stop = private.index("    if (phase_ == QsaCachePhase::Map) {", score_start)
    score = private[score_start:score_stop]
    score = replace_once(score, "4 * kQsaSharedHeadSize", "16 * kQsaSharedHeadSize")
    score = replace_once(score, "head_scores: array<f32, 256>", "head_scores: array<f32, 64>")
    score = replace_once(score, "key_inverse_rms: array<f32, 4>", "key_inverse_rms: array<f32, 16>")
    score = score.replace("group * 64u", "group * 4u").replace("d += 16u", "d += 4u")
    score = replace_once(score, "local_idx / 16u", "local_idx / 4u")
    score = replace_once(score, "local_idx % 16u", "local_idx % 4u")
    score = replace_once(score, "(stride + 3u) / 4u", "(stride + 15u) / 16u")
    score = replace_once(score, "(workgroup_idx % packed_stride) * 4u", "(workgroup_idx % packed_stride) * 16u")
    if pad_keys:
        score = replace_once(score, "16 * kQsaSharedHeadSize", "16 * (kQsaSharedHeadSize + 4)")
        score = score.replace("group * 128u", "group * 132u")
    if cache_query:
        begin = score.index('          << "    let query_base =')
        end = score.index('          << "    var dot =', begin)
        query_rms = score[begin:end]
        score = replace_once(score, query_rms, "")
        score = replace_once(score, "query_value(row, local_lane, d, inv_rms)",
                             "query_values[local_lane * 129u + d]")
        score = replace_once(score,
            '          << "var<workgroup> key_inverse_rms: array<f32, 16>;\\n";',
            '          << "var<workgroup> key_inverse_rms: array<f32, 16>;\\n"\n'
            '          << "var<workgroup> query_inverse_rms: array<f32, 4>;\\n"\n'
            '          << "var<workgroup> query_values: array<f32, " << 4 * (kQsaSharedHeadSize + 1) << ">;\\n";')
        precompute = (
            '          << "  if (local_idx < uniforms.num_heads) {\\n"\n' +
            query_rms.replace("local_lane", "local_idx").replace(
                "let inv_rms =", "query_inverse_rms[local_idx] =") +
            '          << "  }\\n"\n'
            '          << "  workgroupBarrier();\\n"\n'
            '          << "  for (var i = local_idx; i < uniforms.num_heads * uniforms.head_size; i += 64u) {\\n"\n'
            '          << "    let head = i / uniforms.head_size;\\n"\n'
            '          << "    let d = i % uniforms.head_size;\\n"\n'
            '          << "    query_values[head * 129u + d] = query_value(row, head, d, query_inverse_rms[head]);\\n"\n'
            '          << "  }\\n"\n'
            '          << "  workgroupBarrier();\\n"\n')
        anchor = '          << "  let is_valid_block = candidate < visible_count_shared / uniforms.compress_ratio;\\n"\n'
        score = replace_once(score, anchor, precompute + anchor)
    if wide_group:
        score = replace_once(score, "local_idx / 4u", "local_idx / 16u")
        score = replace_once(score, "local_idx % 4u", "local_idx % 16u")
        score = score.replace("d += 4u", "d += 16u")
        score = replace_once(score, "i += 64u", "i += 256u")
        score = replace_once(score, "16 * (kQsaSharedHeadSize + 4)", "16 * (kQsaSharedHeadSize + 16)")
        score = score.replace("group * 132u", "group * 144u")
    private = private[:score_start] + score + private[score_stop:]
    private = private.replace("QsaPackedScoreProgram", "QsaPackedScore16Program").replace(
        'Program{"QsaPackedScore4"}',
        'Program{"QsaPackedScore16QueryWide"}' if wide_group else
        'Program{"QsaPackedScore16QueryCache"}' if cache_query else
        'Program{"QsaPackedScore16Padded"}' if pad_keys else 'Program{"QsaPackedScore16"}')
    source = source[:start] + private + "\n" + source[start:]
    start = source.index(DISPATCH_ANCHOR)
    stop = source.index("    if (use_parallel_rank && phase == QsaCachePhase::Select) {", start)
    dispatch = source[start:stop]
    dispatch = replace_once(dispatch, DISPATCH_ANCHOR,
        "    if (score16 == 1 && (packed_score == 1 || indexer_stack == 1) && use_global_cache && rows == 1 &&\n")
    dispatch = replace_once(dispatch, "num_heads <= 16 && score_stride >= 2048",
        f"num_heads <= 4 && score_stride >= {minimum_stride} &&\n"
        f"        context.DeviceLimits().maxComputeWorkgroupStorageSize >= {storage_bytes}")
    dispatch = replace_once(dispatch, "QsaPackedScoreProgram select", "QsaPackedScore16Program select")
    if wide_group:
        dispatch = replace_once(dispatch, f"maxComputeWorkgroupStorageSize >= {storage_bytes}",
            f"maxComputeWorkgroupStorageSize >= {storage_bytes} &&\n"
            "        context.DeviceLimits().maxComputeWorkgroupSizeX >= 256 &&\n"
            "        context.DeviceLimits().maxComputeInvocationsPerWorkgroup >= 256")
        dispatch = replace_once(dispatch, "SetWorkgroupSize(kQsaSelectWorkgroupSize)", "SetWorkgroupSize(256)")
    dispatch = replace_once(dispatch, "DivideRoundUp(ToUint32(score_stride), 4u)",
                            "DivideRoundUp(ToUint32(score_stride), 16u)")
    source = source[:start] + dispatch + source[start:]
    anchor = '  const int indexer_stack = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QSA_INDEXER_STACK", 0);'
    selector = f'''  const int score16 = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(score16 == 0 || score16 == 1, "{FLAG} must be 0 or 1");
'''
    return replace_once(source, anchor, selector + anchor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, choices=(1, 2, 3, 4, 5), default=2)
    args = parser.parse_args()
    output = OUTPUT.parent / f"build-v{args.version}"
    build_leaf(output=output, source_transform=lambda source: transform(
        source, minimum_stride=8192 if args.version == 2 else 2048,
        pad_keys=args.version >= 3, cache_query=args.version >= 4, wide_group=args.version == 5),
        parent_path=PARENT, component_path=COMPONENT)

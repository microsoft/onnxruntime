from pathlib import Path

from build_qsa_concat_vector import main as build
from build_qmoe_node_replay import replace_once


FLAG = "ORT_WEBGPU_QSA_SMALL_QUERY_CACHE"
PARENT = Path("/workspace/qwen38-int4-qsa-prefill-score16/build-v1")
OUTPUT = Path("/workspace/qwen38-int4-qsa-small-query-cache/build-v2")
STORAGE_BYTES = 4 * 144 * 4 + 16 * 4 + 4 * 4 + 4 + 4 * 129 * 4 + 4 * 4
PROGRAM = "QsaPackedScore4QueryCache"


def cached_program(source):
    start = source.index("class QsaPackedScore16Program final")
    function = source.index("Status QsaPackedScore16Program::GenerateShaderCode", start)
    stop = source.index("\nclass QsaPackedScoreProgram final", function)
    program = source[start:stop].replace("QsaPackedScore16Program", "QsaSmallQueryCacheProgram")
    program = replace_once(program, 'Program{"QsaPackedScore16QueryWide"}', f'Program{{"{PROGRAM}"}}')
    start = program.index("    if (phase_ == QsaCachePhase::Score) {")
    stop = program.index("    if (phase_ == QsaCachePhase::Map) {", start)
    score = program[start:stop]
    for before, after in (
        ("16 * (kQsaSharedHeadSize + 16)", "4 * (kQsaSharedHeadSize + 16)"),
        ("head_scores: array<f32, 64>", "head_scores: array<f32, 16>"),
        ("key_inverse_rms: array<f32, 16>", "key_inverse_rms: array<f32, 4>"),
        ("(stride + 15u) / 16u", "(stride + 3u) / 4u"),
        ("(workgroup_idx % packed_stride) * 16u", "(workgroup_idx % packed_stride) * 4u"),
        ("i += 256u", "i += 64u"),
    ):
        score = replace_once(score, before, after)
    return program[:start] + score + program[stop:]


def transform(source):
    program = cached_program(source)
    start = source.index("class QsaPackedScore16Program final")
    source = source[:start] + program + "\n" + source[start:]
    start = source.index("    if (score16 == 1 && (packed_score == 1 || indexer_stack == 1)")
    stop = source.index("    if ((packed_score == 1 || indexer_stack == 1)", start + 1)
    original = source[start:stop]
    dispatch = replace_once(original, "score16 == 1", "small_query_cache == 1")
    dispatch = replace_once(dispatch, "score_stride >= 2048", "score_stride >= 128 && score_stride < 2048")
    dispatch = replace_once(dispatch, "maxComputeWorkgroupStorageSize >= 11620",
                            f"maxComputeWorkgroupStorageSize >= {STORAGE_BYTES}")
    dispatch = replace_once(dispatch, "maxComputeWorkgroupSizeX >= 256", "maxComputeWorkgroupSizeX >= 64")
    dispatch = replace_once(dispatch, "maxComputeInvocationsPerWorkgroup >= 256",
                            "maxComputeInvocationsPerWorkgroup >= 64")
    dispatch = replace_once(dispatch, "QsaPackedScore16Program select", "QsaSmallQueryCacheProgram select")
    dispatch = replace_once(dispatch, "SetWorkgroupSize(256)", "SetWorkgroupSize(64)")
    dispatch = replace_once(dispatch, "DivideRoundUp(ToUint32(score_stride), 16u)",
                            "DivideRoundUp(ToUint32(score_stride), 4u)")
    source = source[:start] + dispatch + source[start:]
    anchor = '  const int score16 = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QSA_SCORE16", 0);'
    return replace_once(source, anchor,
        f'''  const int small_query_cache = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(small_query_cache == 0 || small_query_cache == 1, "{FLAG} must be 0 or 1");
''' + anchor)


if __name__ == "__main__":
    build(output=OUTPUT, source_transform=transform, parent_path=PARENT, component_path=PARENT)

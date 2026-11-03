from build_qmoe_node_replay import replace_once
from dense_tile_parallel3_transform import PROGRAM as THREE_PROGRAM
from dense_tile_parallel6_long_k_transform import PROGRAM as SIX_PROGRAM, transform as six_transform


FLAG = "ORT_WEBGPU_DENSE_TILE_PARALLEL4_LONG_K"
NAME = "DenseTileParallel4Packed"
SHARED_BYTES = 19456


def make_program():
    source = THREE_PROGRAM.replace("DenseTileParallel3Packed", NAME)
    for before, after in (
        ("tile_a: array<vec4<f16>, 1536>", "tile_a: array<vec4<f16>, 2048>"),
        ("wave * 96u", "wave * 128u"),
        ("wave * 3u + tile_worker", "wave * 4u + tile_worker"),
        ("(wave % 2u) * 3u + tile_worker", "(wave % 2u) * 4u + tile_worker"),
        ("  tile_sums[tile * 128u + logical_idx] = bitcast<u32>(vec2<f16>(sum0, sum1));",
         "  if (tile * 32u < uniforms.blocks) {\n"
         "    tile_sums[tile * 128u + logical_idx] = bitcast<u32>(vec2<f16>(sum0, sum1));\n"
         "  }"),
    ):
        source = replace_once(source, before, after)
    return source


PROGRAM = make_program()


def transform(source):
    source = replace_once(six_transform(source), SIX_PROGRAM, PROGRAM)
    source = source.replace("DenseTileParallel6PackedProgram", "DenseTileParallel4PackedProgram")
    source = source.replace("ORT_WEBGPU_DENSE_TILE_PARALLEL6_LONG_K", FLAG)
    source = source.replace("tile_parallel6", "tile_parallel4")
    for before, after in (
        ("maxComputeWorkgroupSizeX >= 768", "maxComputeWorkgroupSizeX >= 512"),
        ("maxComputeInvocationsPerWorkgroup >= 768", "maxComputeInvocationsPerWorkgroup >= 512"),
        ("maxComputeWorkgroupStorageSize >= 27648", "maxComputeWorkgroupStorageSize >= 19456"),
        (".SetWorkgroupSize(768)", ".SetWorkgroupSize(512)"),
    ):
        source = replace_once(source, before, after)
    return source

from build_qmoe_node_replay import replace_once
from dense_tile_parallel3_transform import PROGRAM as THREE_PROGRAM, transform as three_transform


FLAG = "ORT_WEBGPU_DENSE_TILE_PARALLEL6_LONG_K"
NAME = "DenseTileParallel6Packed"
SHARED_BYTES = 27648


def six_program(source):
    source = source.replace("DenseTileParallel3Packed", NAME)
    for before, after in (
        ("tile_a: array<vec4<f16>, 1536>", "tile_a: array<vec4<f16>, 3072>"),
        ("wave * 96u", "wave * 192u"),
        ("wave * 3u + tile_worker", "wave * 6u + tile_worker"),
        ("(wave % 2u) * 3u + tile_worker", "(wave % 2u) * 6u + tile_worker"),
    ):
        source = replace_once(source, before, after)
    return source


PROGRAM = six_program(THREE_PROGRAM)


def transform(source):
    selected = three_transform(source)
    selected = replace_once(selected, THREE_PROGRAM, PROGRAM)
    selected = selected.replace("DenseTileParallel3PackedProgram", "DenseTileParallel6PackedProgram")
    selected = selected.replace("ORT_WEBGPU_DENSE_TILE_PARALLEL3", FLAG)
    selected = selected.replace("tile_parallel3", "tile_parallel6")
    selected = replace_once(selected, "(K_ == 2560 || K_ == 6144) &&", "N_ == 2560 && K_ == 6144 &&")
    for before, after in (
        ("maxComputeWorkgroupSizeX >= 384", "maxComputeWorkgroupSizeX >= 768"),
        ("maxComputeInvocationsPerWorkgroup >= 384", "maxComputeInvocationsPerWorkgroup >= 768"),
        ("maxComputeWorkgroupStorageSize >= 15360", "maxComputeWorkgroupStorageSize >= 27648"),
        (".SetWorkgroupSize(384)", ".SetWorkgroupSize(768)"),
    ):
        selected = replace_once(selected, before, after)
    return selected

from build_qmoe_node_replay import replace_once
from dense_tile_parallel3_transform import PROGRAM as THREE_PROGRAM, transform as three_transform


FLAG = "ORT_WEBGPU_DENSE_TILE_PARALLEL2"
CLASS = "DenseTileParallel2PackedProgram"
NAME = "DenseTileParallel2Packed"
SHARED_BYTES = 11264


def two_program(program):
    program = program.replace("DenseTileParallel3Packed", NAME)
    for old, new in (("tile_a: array<vec4<f16>, 1536>", "tile_a: array<vec4<f16>, 1024>"),
                     ("wave * 96u < uniforms.blocks", "wave * 64u < uniforms.blocks"),
                     ("wave * 3u + tile_worker", "wave * 2u + tile_worker"),
                     ("(wave % 2u) * 3u + tile_worker", "(wave % 2u) * 2u + tile_worker")):
        program = replace_once(program, old, new)
    return program


PROGRAM = two_program(THREE_PROGRAM)


def transform(source):
    candidate = three_transform(source)
    candidate = replace_once(candidate, THREE_PROGRAM, PROGRAM)
    candidate = candidate.replace("ORT_WEBGPU_DENSE_TILE_PARALLEL3", FLAG).replace(
        "tile_parallel3", "tile_parallel2")
    candidate = replace_once(candidate, "DenseTileParallel3PackedProgram program;", CLASS + " program;")
    for old, new in (("maxComputeWorkgroupSizeX >= 384", "maxComputeWorkgroupSizeX >= 256"),
                     ("maxComputeInvocationsPerWorkgroup >= 384", "maxComputeInvocationsPerWorkgroup >= 256"),
                     ("maxComputeWorkgroupStorageSize >= 15360", "maxComputeWorkgroupStorageSize >= 11264"),
                     (".SetWorkgroupSize(384)", ".SetWorkgroupSize(256)")):
        candidate = replace_once(candidate, old, new)
    return candidate

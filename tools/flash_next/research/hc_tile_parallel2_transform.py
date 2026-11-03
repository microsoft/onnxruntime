from build_qmoe_node_replay import replace_once
from hc_tile_parallel_transform import transform as four_worker_transform


FLAG = "ORT_WEBGPU_HC_TILE_PARALLEL2"
CLASS = "HCPostNormProjectionTileParallel2Program"
NAME = "HCPostNormProjectionTileParallel2F16"
SHARED_BYTES = 37888


def transform(source):
    result = four_worker_transform(source)
    result = result.replace("TILE_PARALLEL4", "TILE_PARALLEL2")
    result = result.replace("TileParallel4", "TileParallel2").replace("tile_parallel4", "tile_parallel2")
    result = replace_once(result, "maxComputeWorkgroupSizeX >= 1024", "maxComputeWorkgroupSizeX >= 512")
    result = replace_once(result, "maxComputeInvocationsPerWorkgroup >= 1024",
                          "maxComputeInvocationsPerWorkgroup >= 512")
    result = replace_once(result, "SetWorkgroupSize(tile_parallel2 == 1 ? 1024 : 256)",
                          "SetWorkgroupSize(tile_parallel2 == 1 ? 512 : 256)")
    return replace_once(result, "tile < 10u; tile += 4u", "tile < 10u; tile += 2u")

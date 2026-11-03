from build_qmoe_node_replay import replace_once
from dense_tile_parallel3_transform import NAME, SHARED_BYTES, transform as broad_transform


FLAG = "ORT_WEBGPU_DENSE_TILE_PARALLEL3_NARROW"


def transform(source):
    result = broad_transform(source).replace("ORT_WEBGPU_DENSE_TILE_PARALLEL3", FLAG)
    return replace_once(result, "(K_ == 2560 || K_ == 6144) &&",
                        "N_ == 2560 && (K_ == 2560 || K_ == 6144) &&")

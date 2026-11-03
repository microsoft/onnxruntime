from build_qmoe_node_replay import replace_once
from dense_tile_parallel2_transform import NAME, PROGRAM, SHARED_BYTES, transform as two_transform


FLAG = "ORT_WEBGPU_DENSE_TILE_PARALLEL2_LONG_K"


def transform(source):
    result = two_transform(source).replace("ORT_WEBGPU_DENSE_TILE_PARALLEL2", FLAG)
    return replace_once(result, "(K_ == 2560 || K_ == 6144) &&",
                         "N_ == 2560 && K_ == 6144 &&")

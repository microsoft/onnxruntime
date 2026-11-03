from build_qmoe_node_replay import replace_once
from dense_tile_parallel3_narrow_transform import NAME, SHARED_BYTES, transform as narrow_transform


FLAG = "ORT_WEBGPU_DENSE_TILE_PARALLEL3_LONG_K"


def transform(source):
    result = narrow_transform(source).replace("ORT_WEBGPU_DENSE_TILE_PARALLEL3_NARROW", FLAG)
    return replace_once(result, "N_ == 2560 && (K_ == 2560 || K_ == 6144) &&",
                        "N_ == 2560 && K_ == 6144 &&")

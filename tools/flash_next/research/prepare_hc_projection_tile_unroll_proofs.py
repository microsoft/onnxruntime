from pathlib import Path

from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent


def base(source):
    source = (source.replace("hc-projection-packed16", "hc-projection-tile-unroll")
              .replace("hc_projection_packed16", "hc_projection_tile_unroll")
              .replace("ORT_WEBGPU_HC_PROJECTION_PACKED16", "ORT_WEBGPU_HC_PROJECTION_TILE_UNROLL")
              .replace("HCProjectionPacked16Shader", "HCProjectionTileUnrollShader"))
    if 'row["logical_streams_per_thread"] == 2' in source:
        source = replace_once(source, 'row["logical_streams_per_thread"] == 2',
                              'row["tile_steps_per_lane"] == 10')
    return source.replace("logical_projection_streams_per_thread=2", "explicit_tile_steps=10")


if __name__ == "__main__":
    for kind in ("native", "model", "screen"):
        target = ROOT / f"run_hc_projection_tile_unroll_{kind}.py"
        with target.open("x") as stream:
            stream.write(base((ROOT / f"run_hc_projection_packed16_{kind}.py").read_text()))
        print(target.name)

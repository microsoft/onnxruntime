from pathlib import Path


ROOT = Path(__file__).parent


def base(source):
    return (source.replace("hc-projection-tile-unroll", "hc-projection-lookahead")
            .replace("hc_projection_tile_unroll", "hc_projection_lookahead")
            .replace("ORT_WEBGPU_HC_PROJECTION_TILE_UNROLL", "ORT_WEBGPU_HC_PROJECTION_LOOKAHEAD")
            .replace("HCProjectionTileUnrollShader", "HCProjectionLookaheadShader")
            .replace('row["tile_steps_per_lane"] == 10', 'row["lookahead_tiles"] == 1')
            .replace("explicit_tile_steps=10", "projection_lookahead_tiles=1")
            .replace('"unroll"', '"lookahead"').replace("'unroll'", "'lookahead'")
            .replace("unroll_decode_gain", "lookahead_decode_gain"))


if __name__ == "__main__":
    for kind in ("native", "model", "screen"):
        target = ROOT / f"run_hc_projection_lookahead_{kind}.py"
        with target.open("x") as stream:
            stream.write(base((ROOT / f"run_hc_projection_tile_unroll_{kind}.py").read_text()))
        print(target.name)

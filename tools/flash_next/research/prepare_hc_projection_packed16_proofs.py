from pathlib import Path


ROOT = Path(__file__).parent


def base(source):
    return (source.replace("hc-projection-lane8", "hc-projection-packed16")
            .replace("hc_projection_lane8", "hc_projection_packed16")
            .replace("ORT_WEBGPU_HC_PROJECTION_LANE8", "ORT_WEBGPU_HC_PROJECTION_PACKED16")
            .replace("HCProjectionLane8Shader", "HCProjectionPacked16Shader")
            .replace('row["columns_per_warp"] == 4', 'row["logical_streams_per_thread"] == 2')
            .replace("projection_columns_per_warp=4", "logical_projection_streams_per_thread=2"))


if __name__ == "__main__":
    for kind in ("native", "model", "screen"):
        target = ROOT / f"run_hc_projection_packed16_{kind}.py"
        with target.open("x") as stream:
            stream.write(base((ROOT / f"run_hc_projection_lane8_{kind}.py").read_text()))
        print(target.name)

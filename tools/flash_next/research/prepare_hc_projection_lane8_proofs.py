from pathlib import Path


ROOT = Path(__file__).parent


def base(source):
    return (source.replace("hc-rms-vector-layout-v1", "hc-projection-lane8-v1")
            .replace("qwen38-int4-hc-rms-vector-layout/build-v1", "qwen38-int4-hc-projection-lane8/build-v1")
            .replace("hc_rms_vector_layout", "hc_projection_lane8")
            .replace("ORT_WEBGPU_HC_RMS_VECTOR_LAYOUT", "ORT_WEBGPU_HC_PROJECTION_LANE8")
            .replace("HCRmsVectorLayoutShader", "HCProjectionLane8Shader")
            .replace('row["components"] == 4', 'row["columns_per_warp"] == 4')
            .replace("rms_shared_vector_components=4", "projection_columns_per_warp=4"))


if __name__ == "__main__":
    for kind in ("native", "model", "screen"):
        target = ROOT / f"run_hc_projection_lane8_{kind}.py"
        with target.open("x") as stream:
            stream.write(base((ROOT / f"run_hc_rms_vector_layout_{kind}.py").read_text()))
        print(target.name)

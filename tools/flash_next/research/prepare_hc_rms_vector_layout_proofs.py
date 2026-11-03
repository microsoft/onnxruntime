from pathlib import Path


ROOT = Path(__file__).parent


def base(source):
    return (source.replace("hc-rms-tree-unroll-v1", "hc-rms-vector-layout-v1")
            .replace("qwen38-int4-hc-rms-tree-unroll/build-v1", "qwen38-int4-hc-rms-vector-layout/build-v1")
            .replace("hc_rms_tree_unroll", "hc_rms_vector_layout")
            .replace("ORT_WEBGPU_HC_RMS_TREE_UNROLL", "ORT_WEBGPU_HC_RMS_VECTOR_LAYOUT")
            .replace("HCRmsTreeUnrollShader", "HCRmsVectorLayoutShader")
            .replace('row["levels"] == 8', 'row["components"] == 4')
            .replace("ordered_reduction_levels=8", "rms_shared_vector_components=4")
            .replace("hc_inner_unroll_four_outputs_finite_bitwise_and_replay_qualified",
                     "hc_rms_vector_layout_four_outputs_finite_bitwise_and_replay_qualified"))


def main():
    for kind in ("native", "model", "screen"):
        target = ROOT / f"run_hc_rms_vector_layout_{kind}.py"
        with target.open("x") as stream:
            stream.write(base((ROOT / f"run_hc_rms_tree_unroll_{kind}.py").read_text()))
        print(target.name)


if __name__ == "__main__":
    main()

from pathlib import Path

from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent


def base(source):
    return (source.replace("hc-rms-tree-unroll-v1", "hc-projection-padding34-v1")
            .replace("qwen38-int4-hc-rms-tree-unroll/build-v1", "qwen38-int4-hc-projection-padding34/build-v1")
            .replace("hc_rms_tree_unroll", "hc_projection_padding34")
            .replace("ORT_WEBGPU_HC_RMS_TREE_UNROLL", "ORT_WEBGPU_HC_PROJECTION_PADDING34")
            .replace("HCRmsTreeUnrollShader", "HCProjectionPaddingShader")
            .replace('row["levels"] == 8', 'row["stride"] == (34 if process_mode == 1 else 36)')
            .replace("ordered_reduction_levels=8", "shared_block_stride_half_elements=34"))


def native_source(source):
    return replace_once(base(source), '"hc_inner_unroll_four_outputs_finite_bitwise_and_replay_qualified"',
                        '"hc_padding34_four_outputs_finite_bitwise_and_replay_qualified"')


def model_source(source):
    source = base(source)
    return replace_once(source, 'row["mode"] == 1 and row["stride"] == (34 if process_mode == 1 else 36)',
                        'row["mode"] == 1 and row["stride"] == 34')


def main():
    for old, new, transform in (
        ("run_hc_rms_tree_unroll_native.py", "run_hc_projection_padding34_native.py", native_source),
        ("run_hc_rms_tree_unroll_model.py", "run_hc_projection_padding34_model.py", model_source),
        ("run_hc_rms_tree_unroll_screen.py", "run_hc_projection_padding34_screen.py", base)):
        with (ROOT / new).open("x") as stream:
            stream.write(transform((ROOT / old).read_text()))
        print(new)


if __name__ == "__main__":
    main()

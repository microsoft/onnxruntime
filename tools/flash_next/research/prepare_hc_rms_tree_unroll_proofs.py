from pathlib import Path

from build_qmoe_node_replay import replace_once
from prepare_hc_projection_inner_unroll_proofs import native_source as inner_native_source


ROOT = Path(__file__).parent


def base(source):
    return (source.replace("hc-projection-inner-unroll-v1", "hc-rms-tree-unroll-v1")
            .replace("qwen38-int4-hc-projection-inner-unroll/build-v1", "qwen38-int4-hc-rms-tree-unroll/build-v1")
            .replace("hc_projection_inner_unroll", "hc_rms_tree_unroll")
            .replace("ORT_WEBGPU_HC_PROJECTION_INNER_UNROLL_STATS_FILE", "ORT_WEBGPU_HC_RMS_TREE_UNROLL_STATS_FILE")
            .replace("HCProjectionInnerUnrollShader", "HCRmsTreeUnrollShader")
            .replace('row["component_steps"] == 4', 'row["levels"] == 8')
            .replace("ordered_component_steps=4", "ordered_reduction_levels=8"))


def native_source(source):
    source = base(inner_native_source(source))
    return replace_once(source, "    evidence = [build_path, Path(__file__),",
        '''    evidence = [ROOT / "build_scoped_webgpu_source.py",
                ROOT / "build_hc_projection_inner_unroll.py",
                ROOT / "prepare_hc_projection_inner_unroll_proofs.py",
                build_path, Path(__file__),''')


def model_source(source):
    source = base(source)
    source = replace_once(source, "    return dict(actual_dense_replay_instances=len(dense),",
        '''    hc = [row for row in rows if row.get("operator") in
          ("HCPostNormProjectionF16", "HCSharedPostNormProjectionF16")]
    assert len(hc) == 94, len(hc)
    assert all(row["hits"] > 0 and row["binding_hits"] > 0 and
               row["rebinds"] + row["binding_hits"] == row["hits"] for row in hc)
    return dict(actual_hc_replay_instances=len(hc),
                hc_replay_hits=sum(row["hits"] for row in hc),
                hc_binding_hits=sum(row["binding_hits"] for row in hc),
                actual_dense_replay_instances=len(dense),''')
    return source


def main():
    for old, new, transform in (
        ("run_hc_bitcast_native.py", "run_hc_rms_tree_unroll_native.py", native_source),
        ("run_hc_projection_inner_unroll_model.py", "run_hc_rms_tree_unroll_model.py", model_source),
        ("run_hc_projection_inner_unroll_screen.py", "run_hc_rms_tree_unroll_screen.py", base)):
        target = ROOT / new
        with target.open("x") as stream:
            stream.write(transform((ROOT / old).read_text()))
        print(target.name)


if __name__ == "__main__":
    main()

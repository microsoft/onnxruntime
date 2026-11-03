from pathlib import Path

from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent


def transform(source):
    source = source.replace("from run_dense_output_shape_native import", "from run_dense_output_shape_static_native import")
    source = source.replace("dense-output-shape-v2", "dense-output-shape-static-v1")
    source = source.replace("qwen38-int4-dense-output-shape/build-v2", "qwen38-int4-dense-output-shape-static/build-v1")
    source = source.replace("dense_output_shape_changing_inputs_finite_bitwise_qualified",
                            "dense_output_shape_session_static_changing_inputs_finite_bitwise_qualified")
    source = source.replace("MatMulNBitsOutputShapeExact", "MatMulNBitsStaticOutputShapeExact")
    source = replace_once(source, "    assert len(rows) == 36", "    assert len(rows) == 72")
    source = source.replace('r["requests"] == 256', 'r["requests"] == 128')
    source = source.replace('r["hits"] == 255', 'r["hits"] == 127')
    source = replace_once(source,
        '    assert len(shapes) == 36 and sum(r["fast_shape_hits"] == 128 for r in shapes) == 27\n'
        '    assert sum(r["fast_shape_hits"] == 0 for r in shapes) == 9',
        '    assert len(shapes) == 72 and sum(r["fast_shape_hits"] == 128 for r in shapes) == 27\n'
        '    assert sum(r["fast_shape_hits"] == 0 for r in shapes) == 45')
    source = source.replace("dense_output_shape_rank1_fallback_rank234_finite_bitwise_qualified",
                            "dense_output_shape_session_static_rank1_fallback_rank234_finite_bitwise_qualified")
    source = replace_once(source, "rank1_fallback_fixtures=9,", "rank1_fallback_fixtures=9, control_shape_kernels=36,")
    return source


def main():
    target = ROOT / "run_dense_output_shape_static_ranks.py"
    if target.exists():
        raise FileExistsError(target)
    with target.open("x") as stream:
        stream.write(transform((ROOT / "run_dense_output_shape_ranks.py").read_text()))
    print(target.name)


if __name__ == "__main__":
    main()

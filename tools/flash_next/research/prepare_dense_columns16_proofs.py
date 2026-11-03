from pathlib import Path

from prepare_dense_paired_matrix_reset_proofs import make_runner as synthetic_runner
from prepare_dense_paired_matrix_actual_proofs import make_runner as actual_runner


ROOT = Path(__file__).parent


def adapt(source):
    for before, after in (
        ("build_dense_paired_matrix_reset", "build_dense_columns16"),
        ("test_dense_paired_matrix_reset.py", "test_dense_columns16.py"),
        ("qwen38-int4-dense-paired-matrix-reset", "qwen38-int4-dense-columns16"),
        ("dense-paired-matrix-reset-", "dense-columns16-"),
        ("packed_dense_original_provider_synthetic_shapes_supported",
         "dense_columns16_original_provider_synthetic_shapes_supported"),
        ("packed_dense_original_provider_actual_weights_supported",
         "dense_columns16_original_provider_actual_weights_supported"),
        ("DensePairedMatrixReset", "DenseColumns16"),
        ("native_events", "selected_events"),
        ("native_dispatches", "selected_dispatches"),
        ("Native matrix dispatch coverage", "Selected kernel dispatch coverage"),
        ("Unexpected native matrix", "Unexpected selected kernel"),
    ):
        source = source.replace(before, after)
    return source


if __name__ == "__main__":
    for name, source in (
        ("run_dense_columns16_synthetic.py", synthetic_runner()),
        ("run_dense_columns16_actual.py", actual_runner()),
    ):
        target = ROOT / name
        with target.open("x") as stream:
            stream.write(adapt(source))
        print(target.name)

import ast
from pathlib import Path

from prepare_dense_columns8_word_unroll_proofs import batch_source as parent_source


ROOT = Path(__file__).parent


def batch_source(actual=False):
    source = parent_source(actual)
    for before, after in (("dense-columns8-word-unroll", "dense-columns8-private-totals"),
                          ("ORT_WEBGPU_DENSE_COLUMNS8_WORD_UNROLL", "ORT_WEBGPU_DENSE_COLUMNS8_PRIVATE_TOTALS"),
                          ("DenseColumns8WordUnroll", "DenseColumns8PrivateTotals"),
                          ("dense_columns8_word_unroll", "dense_columns8_private_totals"),
                          ("columns8_word_unroll_", "columns8_private_totals_")):
        source = source.replace(before, after)
    ast.parse(source)
    return source


if __name__ == "__main__":
    for actual in (False, True):
        name = ("run_dense_columns8_private_totals_actual_batch.py" if actual
                else "run_dense_columns8_private_totals_batch.py")
        with (ROOT / name).open("x") as stream:
            stream.write(batch_source(actual))
        print(name)

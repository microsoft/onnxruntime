import ast
from pathlib import Path

from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent


def batch_source(actual=False):
    parent_name = ("run_dense_columns8_double_buffer_actual_batch.py" if actual
                   else "run_dense_columns8_double_buffer_batch.py")
    source = (ROOT / parent_name).read_text()
    for before, after in (
        ("dense-columns8-double-buffer", "dense-columns8-word-unroll"),
        ("ORT_WEBGPU_DENSE_COLUMNS8_DOUBLE_BUFFER", "ORT_WEBGPU_DENSE_COLUMNS8_WORD_UNROLL"),
        ("DenseColumns8DoubleBuffer", "DenseColumns8WordUnroll"),
        ("columns8_double_buffer_", "columns8_word_unroll_"),
        ("build_dense_columns8_double_buffer.py", "build_dense_columns8_word_unroll.py"),
        ("test_dense_columns8_double_buffer.py", "test_dense_columns8_word_unroll.py"),
        ("prepare_dense_columns8_double_buffer_proofs.py", "prepare_dense_columns8_word_unroll_proofs.py"),
    ):
        source = source.replace(before, after)
    source = replace_once(source, '                ROOT / "prepare_dense_columns8_double_buffer.py",',
                          f'                ROOT / "{parent_name}",')
    source = replace_once(source, "ROOT = Path(__file__).parent",
                          'FLAGS += " ORT_WEBGPU_DENSE_COLUMNS8_DOUBLE_BUFFER=1"\n\nROOT = Path(__file__).parent')
    source = replace_once(source, '            if any(e.get("dur", 0) <= 0 for e in dispatched):',
        '''            parent_chosen = sum("DenseColumns8DoubleBuffer" in e.get("name", "") for e in dispatched)
            if parent_chosen != (64 if phase == "default" and n >= 2560 else 0):
                raise RuntimeError("Missing supported-parent/default dispatch proof")
            if any(e.get("dur", 0) <= 0 for e in dispatched):''')
    ast.parse(source)
    return source


if __name__ == "__main__":
    for actual in (False, True):
        name = ("run_dense_columns8_word_unroll_actual_batch.py" if actual
                else "run_dense_columns8_word_unroll_batch.py")
        with (ROOT / name).open("x") as stream:
            stream.write(batch_source(actual))
        print(name)

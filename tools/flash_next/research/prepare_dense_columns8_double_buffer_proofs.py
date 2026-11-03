import ast
from pathlib import Path

from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent


def batch_source(actual=False):
    parent_name = "run_dense_columns8_actual_batch.py" if actual else "run_dense_columns8_batch.py"
    source = (ROOT / parent_name).read_text()
    for before, after in (
        ("dense-columns8", "dense-columns8-double-buffer"),
        ("ORT_WEBGPU_DENSE_COLUMNS8", "ORT_WEBGPU_DENSE_COLUMNS8_DOUBLE_BUFFER"),
        ("DenseColumns8", "DenseColumns8DoubleBuffer"),
        ("columns8_", "columns8_double_buffer_"),
        ("build_dense_columns8.py", "build_dense_columns8_double_buffer.py"),
        ("prepare_dense_columns8.py", "prepare_dense_columns8_double_buffer.py"),
        ("test_dense_columns8.py", "test_dense_columns8_double_buffer.py"),
    ):
        if before not in source:
            raise ValueError(f"Missing parent batch marker: {before}")
        source = source.replace(before, after)
    source = replace_once(
        source, "parent_path, build_path, provider, ORIGINAL, binary,",
        f'ROOT / "prepare_dense_columns8_double_buffer_proofs.py",\n'
        f'                ROOT / "{parent_name}",\n'
        "                parent_path, build_path, provider, ORIGINAL, binary,")
    ast.parse(source)
    return source


if __name__ == "__main__":
    for actual in (False, True):
        name = ("run_dense_columns8_double_buffer_actual_batch.py" if actual
                else "run_dense_columns8_double_buffer_batch.py")
        with (ROOT / name).open("x") as stream:
            stream.write(batch_source(actual))
        print(name)

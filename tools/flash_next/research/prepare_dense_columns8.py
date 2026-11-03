from pathlib import Path

from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent


def builder_source():
    source = (ROOT / "build_dense_columns16.py").read_text()
    for before, after in (
        ("COLUMNS16", "COLUMNS8"), ("Columns16", "Columns8"),
        ("columns16", "columns8"), ("range(4)", "range(2)"),
        ("STORAGE_BYTES = 3072", "STORAGE_BYTES = 2560"),
        ("partials: array<f16, 512>", "partials: array<f16, 256>"),
        ("i < 512u", "i < 256u"), ("workgroup_idx * 16u", "workgroup_idx * 8u"),
        ("local_idx < 16u", "local_idx < 8u"),
        ("N_ % 16 == 0", "N_ >= 2560 && N_ % 8 == 0"),
        ("N_ / 16", "N_ / 8"), ("columns8 ? 5", "columns8 ? 6"),
        ("output_columns=16, columns_per_thread=4", "output_columns=8, columns_per_thread=2"),
    ):
        source = source.replace(before, after)
    return source


if __name__ == "__main__":
    target = ROOT / "build_dense_columns8.py"
    with target.open("x") as stream:
        stream.write(builder_source())
    print(target.name)

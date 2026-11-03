from pathlib import Path

from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent


def builder_source():
    source = (ROOT / "build_dense_columns8.py").read_text()
    if source.count("DenseColumns8") != 5:
        raise ValueError("Unexpected parent program names")
    source = source.replace("DenseColumns8", "DenseColumns8DoubleBuffer")
    for before, after in (
        ("ORT_WEBGPU_DENSE_COLUMNS8", "ORT_WEBGPU_DENSE_COLUMNS8_DOUBLE_BUFFER"),
        ("qwen38-int4-dense-columns8/", "qwen38-int4-dense-columns8-double-buffer/"),
        ("STORAGE_BYTES = 2560", "STORAGE_BYTES = 4608"),
        ("tile_a: array<vec4<f16>, 256>", "tile_a: array<vec4<f16>, 512>"),
        ("tile_a[i] = value;", "tile_a[(tile % 2u) * 256u + i] = value;"),
        ("tile_a[stream * 8u + word * 2u]",
         "tile_a[(tile % 2u) * 256u + stream * 8u + word * 2u]"),
        ("tile_a[stream * 8u + word * 2u + 1u]",
         "tile_a[(tile % 2u) * 256u + stream * 8u + word * 2u + 1u]"),
        ("  workgroupBarrier();\n}\nif (local_idx < 8u)",
         "}\nworkgroupBarrier();\nif (local_idx < 8u)"),
        ("columns8 ? 6", "columns8 ? 7"),
        ("original_k_streams=32, output_columns=8, columns_per_thread=2,",
         "original_k_streams=32, output_columns=8, columns_per_thread=2,\n"
         "             activation_buffers=2, barriers_per_k_tile=1, final_partial_barrier=True,"),
    ):
        source = replace_once(source, before, after)
    return source


if __name__ == "__main__":
    target = ROOT / "build_dense_columns8_double_buffer.py"
    with target.open("x") as stream:
        stream.write(builder_source())
    print(target.name)

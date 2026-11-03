import ast
from pathlib import Path

from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent


if __name__ == "__main__":
    source = (ROOT / "run_dense_columns16_spatial_double_buffer_unprofiled.py").read_text()
    for before, after in (
        ("dense-columns16-spatial-double-buffer", "dense-columns8-spatial-double-buffer"),
        ("dense_columns16_spatial_double_buffer", "dense_columns8_spatial_double_buffer"),
        ("columns16_spatial_double_buffer_", "columns8_spatial_double_buffer_"),
        ("-gpu7", "-gpu4"),
    ):
        source = source.replace(before, after)
    source = replace_once(source,
        "from run_dense_columns16_synthetic import FLAGS as PARENT_FLAGS, UUID, run_guard, workspace",
        "from run_dense_columns16_synthetic import FLAGS as PARENT_FLAGS, workspace\n"
        "from run_dense_columns8_spatial_double_buffer_batch import UUID, run_guard")
    source = replace_once(source,
        'FLAGS = PARENT_FLAGS + " ORT_WEBGPU_DENSE_COLUMNS8_DOUBLE_BUFFER=1"',
        'FLAGS = PARENT_FLAGS + (" ORT_WEBGPU_DENSE_COLUMNS8_DOUBLE_BUFFER=1"\n'
        '                        " ORT_WEBGPU_SMALL_UPLOAD_WRITE_BUFFER=1"\n'
        '                        " ORT_WEBGPU_SMALL_READBACK_CACHE=1"\n'
        '                        " ORT_WEBGPU_UNIFORM_ARENA=1")')
    source = replace_once(source, "profiling=False, timings_include_host_run_and_readback=True,",
        "profiling=False, timings_include_host_run_and_readback=True,\n"
        '                  small_upload_write_buffer=True, small_readback_cache=True, uniform_arena=True,')
    source = replace_once(source, "evidence = [Path(__file__), actual_path, manifest_path, provider, binary]",
        'evidence = [Path(__file__), ROOT / "prepare_dense_columns8_spatial_unprofiled.py",\n'
        "                actual_path, manifest_path, provider, binary]")
    ast.parse(source)
    output = ROOT / "run_dense_columns8_spatial_double_buffer_unprofiled.py"
    with output.open("x") as stream:
        stream.write(source)
    print(output.name)

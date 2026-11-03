import ast
from pathlib import Path

from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent


def adapt(source):
    source = source.replace("dense-columns8", "dense-columns8-double-buffer")
    source = source.replace("DenseColumns8", "DenseColumns8DoubleBuffer")
    source = source.replace("columns8_", "columns8_double_buffer_")
    source = source.replace("build_dense_columns8 import", "build_dense_columns8_double_buffer import")
    return source.replace('STAMP = "20261031"', 'STAMP = "20261101"')


def model_source():
    source = adapt((ROOT / "run_dense_columns8_model.py").read_text())
    return replace_once(
        source, 'sources = [Path(__file__), ROOT / "benchmark_webgpu_int4.py",',
        'sources = [Path(__file__), ROOT / "prepare_dense_columns8_double_buffer_model.py",\n'
        '               ROOT / "benchmark_webgpu_int4.py",')


def invalid_source():
    return adapt((ROOT / "run_dense_columns8_invalid.py").read_text())


def standard_source():
    source = adapt((ROOT / "run_dense_columns8_cpu_local8_standard.py").read_text())
    source = replace_once(
        source,
        '    gpu_path = ROOT / "dense-columns8-double-buffer-v1-context2048-repeat-screen-20261031.qualification.json"\n'
        '    gpu = json.loads(gpu_path.read_text())\n'
        '    assert gpu["status"] == "repeated_columns8_double_buffer_decode_gain_both_modes"',
        '    gpu_path = ROOT / "dense-columns8-double-buffer-v1-columns8-gpu7-full-profile2048-trial0-20261101.qualification.json"\n'
        '    gpu = json.loads(gpu_path.read_text())\n'
        '    assert gpu["status"] == "original_sequences_qualified"\n'
        '    assert gpu["phase"] == "profile" and gpu["context"] == 2048\n'
        '    assert gpu["mode"] == "columns8" and gpu["full_validation"]\n'
        '    assert gpu["experiment_flag"] == FLAG and gpu["experiment_flag_value"] == 1\n'
        '    assert gpu["result"]["program_counts"]["DenseColumns8DoubleBuffer"] == 158\n'
        '    assert gpu["result"]["program_counts"]["MatMulNBitsBlock32F16"] == 1')
    return source


def screen_source():
    return adapt((ROOT / "run_dense_columns8_cpu_local8_screen.py").read_text())


if __name__ == "__main__":
    for name, source in (
        ("run_dense_columns8_double_buffer_invalid.py", invalid_source()),
        ("run_dense_columns8_double_buffer_model.py", model_source()),
        ("run_dense_columns8_double_buffer_cpu_local8_standard.py", standard_source()),
        ("run_dense_columns8_double_buffer_cpu_local8_screen.py", screen_source()),
    ):
        ast.parse(source)
        with (ROOT / name).open("x") as stream:
            stream.write(source)
        print(name)

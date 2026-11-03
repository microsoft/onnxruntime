from pathlib import Path

from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent


def write_new(name, text):
    with (ROOT / name).open("x") as stream:
        stream.write(text)


if __name__ == "__main__":
    source = (ROOT / "argmax_parallel_compare.cpp").read_text()
    source = replace_once(
        source, "    const auto info = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();",
        "    const auto input_type = session.GetInputTypeInfo(0);\n"
        "    const auto info = input_type.GetTensorTypeAndShapeInfo();")
    write_new("argmax_parallel_compare_v3.cpp", source)
    builder = (ROOT / "build_argmax_parallel.py").read_text()
    builder = replace_once(builder, "argmax-parallel/build-v2", "argmax-parallel/build-v3")
    builder = replace_once(builder, 'ROOT / "argmax_parallel_compare.cpp"', 'ROOT / "argmax_parallel_compare_v3.cpp"')
    write_new("build_argmax_parallel_v3.py", builder)
    runner = (ROOT / "run_argmax_parallel_primitive.py").read_text()
    runner = runner.replace("from build_argmax_parallel import", "from build_argmax_parallel_v3 import")
    runner = runner.replace("build-v2", "build-v3").replace("fixtures-v2", "fixtures-v3")
    runner = runner.replace("argmax-parallel-v2", "argmax-parallel-v3")
    runner = runner.replace('"build_argmax_parallel.py"', '"build_argmax_parallel_v3.py"')
    runner = runner.replace('"argmax_parallel_compare.cpp"', '"argmax_parallel_compare_v3.cpp"')
    # The immutable original provider remains the 0955 rows4 build-v2.
    runner = replace_once(runner, "qmoe-prefill-rows4/build-v3", "qmoe-prefill-rows4/build-v2")
    write_new("run_argmax_parallel_primitive_v3.py", runner)

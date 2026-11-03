from pathlib import Path

from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent


if __name__ == "__main__":
    source = (ROOT / "argmax_parallel_compare_v3.cpp").read_text()
    source = replace_once(source, "bool timing) {", "bool timing, bool exclude_subnormals) {")
    source = replace_once(source, "    const int pattern = timing ? 0 : round;",
                          "    const int pattern = timing || (exclude_subnormals && round == 12) ? 0 : round;")
    source = replace_once(source, 'std::string(argv[1]) != "timing"))',
                          'std::string(argv[1]) != "timing" && std::string(argv[1]) != "compare-normal"))')
    source = source.replace("session, shape, timing);",
                            'session, shape, timing, std::string(argv[1]) == "compare-normal");')
    source = source.replace("Usage: compare|timing provider", "Usage: compare|compare-normal|timing provider")
    with (ROOT / "argmax_parallel_compare_bitwise.cpp").open("x") as stream:
        stream.write(source)
    runner = (ROOT / "run_argmax_parallel_primitive_v3.py").read_text()
    runner = runner.replace("from build_argmax_parallel_v3 import", "from build_argmax_parallel_bitwise import")
    runner = replace_once(runner, 'BUILD = Path("/home/sayanshaw/repos/qwen38-int4-argmax-parallel/build-v3")',
                          'BUILD = Path("/home/sayanshaw/repos/qwen38-int4-argmax-parallel/build-v4")')
    runner = runner.replace("fixtures-v3", "fixtures-v4").replace("argmax-parallel-v3", "argmax-parallel-bitwise-v4")
    runner = runner.replace('BUILD / "argmax_compare"', 'BUILD / "bitwise_argmax_compare"')
    runner = runner.replace('manifest["comparator_sha256"]', 'manifest["bitwise_comparator_sha256"]')
    runner = runner.replace('"build_argmax_parallel_v3.py"', '"build_argmax_parallel_bitwise.py"')
    runner = runner.replace('"argmax_parallel_compare_v3.cpp"', '"argmax_parallel_compare_bitwise.cpp"')
    runner = replace_once(runner, 'workspace(binary), "compare",',
                          'workspace(binary), "compare" if mode == 1 and n >= 32 else "compare-normal",')
    runner = replace_once(runner, "nonfinite_inputs_signed_zero_subnormals_and_ties_included=True,",
        "nonfinite_inputs_signed_zero_subnormals_and_ties_included=True,\n"
        "                  selected_eligible_shapes_include_subnormals=True,\n"
        "                  original_default_and_small_fallback_exclude_known_subnormal_divergence=True,")
    with (ROOT / "run_argmax_parallel_bitwise.py").open("x") as stream:
        stream.write(runner)

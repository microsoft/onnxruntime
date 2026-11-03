import json
from pathlib import Path

from build_qmoe_node_replay import replace_once, sha


ROOT = Path(__file__).parent


def native_source(source):
    return replace_once(
        source,
        "shape.size() != 2 || output_shape.size() != 2 || shape[0] <= 0 || shape[1] <= 0 ||\n"
        "          output_shape[0] != shape[0] || output_shape[1] <= 0",
        "shape.size() < 2 || output_shape.size() != shape.size() ||\n"
        "          !std::equal(shape.begin(), shape.end() - 1, output_shape.begin()) ||\n"
        "          std::any_of(shape.begin(), shape.end(), [](int64_t dimension) { return dimension <= 0; }) ||\n"
        "          std::any_of(output_shape.begin(), output_shape.end(), [](int64_t dimension) { return dimension <= 0; })")


def build_source(source):
    source = source.replace("native-full-v1", "native-full-v2")
    source = source.replace("dense_long_k_fixed_native.cpp", "dense_long_k_fixed_native_v2.cpp")
    return replace_once(
        source, "(Path(__file__), source)",
        '(Path(__file__), source, ROOT / "dense_long_k_fixed_native.cpp",\n'
        '                                                                  ROOT / "build_dense_long_k_native.py",\n'
        '                                                                  ROOT / "prepare_dense_long_k_native_v2.py")')


def controller_source(source):
    return source.replace("native-full-v1", "native-full-v2").replace(
        "native-proof-v1", "native-proof-v2").replace(
        'VARIANT = "dense-tile-parallel3-long-k-native-v1"',
        'VARIANT = "dense-tile-parallel3-long-k-native-v2"')


def main():
    parents = (ROOT / "dense_long_k_fixed_native.cpp", ROOT / "build_dense_long_k_native.py",
               ROOT / "run_dense_long_k_native_v1.py")
    outputs = (ROOT / "dense_long_k_fixed_native_v2.cpp", ROOT / "build_dense_long_k_native_v2.py",
               ROOT / "run_dense_long_k_native_v2.py")
    report = ROOT / "dense-long-k-native-v2-generation.json"
    if report.exists() or any(path.exists() for path in outputs):
        raise FileExistsError("Fresh rank-aware native generation required")
    for parent, output, transform in zip(parents, outputs,
                                         (native_source, build_source, controller_source), strict=True):
        text = transform(parent.read_text())
        if output.suffix == ".py":
            compile(text, str(output), "exec")
        with output.open("x") as stream:
            stream.write(text)
    with report.open("x") as stream:
        json.dump({"status": "generated_not_qualified",
                   "reason": "Frozen native fixtures are rank3 [1,rows,K], not rank2; preserve all types/counts/shape equality.",
                   "source_sha256": {str(path): sha(path) for path in (Path(__file__), *parents, *outputs)}},
                  stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()

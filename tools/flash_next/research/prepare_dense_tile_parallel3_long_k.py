import json
from pathlib import Path

from build_qmoe_node_replay import replace_once, sha
from prepare_dense_tile_parallel3_narrow import build_source as narrow_build_source


ROOT = Path(__file__).parent


def build_source(source):
    source = narrow_build_source(source)
    source = source.replace("qwen38-int4-dense-tile-parallel3-narrow/",
                            "qwen38-int4-dense-tile-parallel3-long-k/")
    source = source.replace("dense_tile_parallel3_narrow_transform", "dense_tile_parallel3_long_k_transform")
    source = source.replace("prepare_dense_tile_parallel3_narrow.py", "prepare_dense_tile_parallel3_long_k.py")
    source = replace_once(source, '"supported_k": [2560, 6144],', '"supported_k": [6144],')
    return replace_once(
        source, "        Path(__file__),",
        '        Path(__file__), ROOT / "dense_tile_parallel3_narrow_transform.py",\n'
        '        ROOT / "prepare_dense_tile_parallel3_narrow.py",')


def main():
    parent = ROOT / "build_dense_tile_parallel3.py"
    output = ROOT / "build_dense_tile_parallel3_long_k_v1.py"
    report = ROOT / "dense-tile-parallel3-long-k-v1-generation.json"
    if output.exists() or report.exists():
        raise FileExistsError("Fresh long-K generation required")
    source = build_source(parent.read_text())
    compile(source, str(output), "exec")
    with output.open("x") as stream:
        stream.write(source)
    with report.open("x") as stream:
        json.dump({"status": "generated_not_qualified",
                   "source_sha256": {str(path): sha(path) for path in (
                       Path(__file__), ROOT / "prepare_dense_tile_parallel3_narrow.py",
                       ROOT / "dense_tile_parallel3_long_k_transform.py",
                       ROOT / "dense_tile_parallel3_narrow_transform.py",
                       ROOT / "dense_tile_parallel3_transform.py", parent, output)}}, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()

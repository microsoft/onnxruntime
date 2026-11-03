import json
from pathlib import Path

from build_qmoe_node_replay import replace_once, sha


ROOT = Path(__file__).parent


def build_source(source):
    source = source.replace("qwen38-int4-dense-tile-parallel3/", "qwen38-int4-dense-tile-parallel3-narrow/")
    source = source.replace("from dense_tile_parallel3_transform import",
                            "from dense_tile_parallel3_narrow_transform import")
    source = source.replace('ROOT / "dense_tile_parallel3_transform.py"',
                            'ROOT / "dense_tile_parallel3_narrow_transform.py"')
    source = replace_once(source, "        Path(__file__), ROOT /",
                          '        Path(__file__), ROOT / "dense_tile_parallel3_transform.py",\n'
                          '        ROOT / "build_dense_tile_parallel3.py",\n'
                          '        ROOT / "prepare_dense_tile_parallel3_narrow.py", ROOT /')
    return replace_once(source, '"supported_k": [2560, 6144],',
                        '"supported_k": [2560, 6144], "supported_n": 2560,')


def controller_source(source):
    source = source.replace("qwen38-int4-dense-tile-parallel3/", "qwen38-int4-dense-tile-parallel3-narrow/")
    source = source.replace("from dense_tile_parallel3_transform import",
                            "from dense_tile_parallel3_narrow_transform import")
    source = source.replace('VARIANT = "dense-tile-parallel3-v1"',
                            'VARIANT = "dense-tile-parallel3-narrow-v1"')
    source = source.replace('ROOT / "dense_tile_parallel3_transform.py"',
                            'ROOT / "dense_tile_parallel3_narrow_transform.py"')
    source = replace_once(source, "bool(selected and k in (2560, 6144))",
                          "bool(selected and n == 2560 and k in (2560, 6144))")
    return replace_once(source, '"selected": k in (2560, 6144),',
                        '"selected": n == 2560 and k in (2560, 6144),')


def main():
    parents = (ROOT / "build_dense_tile_parallel3.py", ROOT / "run_dense_tile_parallel3_v1.py")
    outputs = (ROOT / "build_dense_tile_parallel3_narrow_v1.py",
               ROOT / "run_dense_tile_parallel3_narrow_v1.py")
    report = ROOT / "dense-tile-parallel3-narrow-v1-generation.json"
    if report.exists() or any(path.exists() for path in outputs):
        raise FileExistsError("Fresh narrow dense generation required")
    for parent, output, transform in zip(parents, outputs, (build_source, controller_source), strict=True):
        text = transform(parent.read_text())
        compile(text, str(output), "exec")
        with output.open("x") as stream:
            stream.write(text)
    with report.open("x") as stream:
        json.dump({"status": "generated_not_qualified",
                   "source_sha256": {str(path): sha(path) for path in (
                       Path(__file__), ROOT / "dense_tile_parallel3_transform.py",
                       ROOT / "dense_tile_parallel3_narrow_transform.py", *parents, *outputs)}},
                  stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()

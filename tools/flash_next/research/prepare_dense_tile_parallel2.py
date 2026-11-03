import json
from pathlib import Path

from build_qmoe_node_replay import replace_once, sha


ROOT = Path(__file__).parent


def build_source(source):
    source = source.replace("qwen38-int4-dense-tile-parallel3/", "qwen38-int4-dense-tile-parallel2/")
    source = source.replace("from dense_tile_parallel3_transform import", "from dense_tile_parallel2_transform import")
    source = source.replace('ROOT / "dense_tile_parallel3_transform.py"', 'ROOT / "dense_tile_parallel2_transform.py"')
    source = replace_once(source, "        Path(__file__), ROOT /",
                          '        Path(__file__), ROOT / "dense_tile_parallel3_transform.py",\n'
                          '        ROOT / "build_dense_tile_parallel3.py",\n'
                          '        ROOT / "prepare_dense_tile_parallel2.py", ROOT /')
    return replace_once(source, '"physical_threads": 384, "independent_tile_workers": 3,',
                        '"physical_threads": 256, "independent_tile_workers": 2,')


def controller_source(source):
    source = source.replace("qwen38-int4-dense-tile-parallel3/", "qwen38-int4-dense-tile-parallel2/")
    source = source.replace("from dense_tile_parallel3_transform import", "from dense_tile_parallel2_transform import")
    source = source.replace('ROOT / "dense_tile_parallel3_transform.py"', 'ROOT / "dense_tile_parallel2_transform.py"')
    source = replace_once(source, 'VARIANT = "dense-tile-parallel3-v1"', 'VARIANT = "dense-tile-parallel2-v1"')
    source = replace_once(source, "threads = 384 if selected else 128", "threads = 256 if selected else 128")
    source = replace_once(source, 'manifest["physical_threads"] != 384', 'manifest["physical_threads"] != 256')
    return replace_once(source, 'manifest["shared_storage_bytes"] != 15360',
                        'manifest["shared_storage_bytes"] != 11264')


def main():
    parents = (ROOT / "build_dense_tile_parallel3.py", ROOT / "run_dense_tile_parallel3_v1.py")
    outputs = (ROOT / "build_dense_tile_parallel2_v1.py", ROOT / "run_dense_tile_parallel2_v1.py")
    report = ROOT / "dense-tile-parallel2-v1-generation.json"
    if report.exists() or any(path.exists() for path in outputs):
        raise FileExistsError("Fresh two-worker dense generation required")
    for parent, output, transform in zip(parents, outputs, (build_source, controller_source), strict=True):
        source = transform(parent.read_text())
        compile(source, str(output), "exec")
        with output.open("x") as stream:
            stream.write(source)
    with report.open("x") as stream:
        json.dump({"status": "generated_not_qualified",
                   "reason": "Three-worker broad route regressed wide shapes; test a distinct 256-thread/11264-byte two-worker geometry without changing half dot/fold order.",
                   "source_sha256": {str(path): sha(path) for path in
                                     (Path(__file__), ROOT / "dense_tile_parallel3_transform.py",
                                      ROOT / "dense_tile_parallel2_transform.py", *parents, *outputs)}},
                  stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()

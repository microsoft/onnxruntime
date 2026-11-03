import json
from pathlib import Path

from build_qmoe_node_replay import replace_once, sha


ROOT = Path(__file__).parent


def build_source(source):
    source = source.replace("qwen38-int4-dense-tile-parallel2/", "qwen38-int4-dense-tile-parallel2-long-k/")
    source = source.replace("from dense_tile_parallel2_transform import",
                            "from dense_tile_parallel2_long_k_transform import")
    source = replace_once(
        source, 'ROOT / "dense_tile_parallel2_transform.py",',
        'ROOT / "dense_tile_parallel2_transform.py", ROOT / "dense_tile_parallel2_long_k_transform.py",\n'
        '        ROOT / "prepare_dense_tile_parallel2_long_k.py",')
    return replace_once(source, '"supported_k": [2560, 6144],',
                         '"supported_k": [6144], "supported_n": [2560],')


def controller_source(source):
    source = source.replace("qwen38-int4-dense-tile-parallel2/", "qwen38-int4-dense-tile-parallel2-long-k/")
    source = source.replace("from dense_tile_parallel2_transform import",
                            "from dense_tile_parallel2_long_k_transform import")
    source = source.replace('VARIANT = "dense-tile-parallel2-v1"',
                            'VARIANT = "dense-tile-parallel2-long-k-v1"')
    source = replace_once(source, "bool(selected and k in (2560, 6144))",
                          "bool(selected and (k, n) == (6144, 2560))")
    source = replace_once(source, '"selected": k in (2560, 6144)',
                          '"selected": (k, n) == (6144, 2560)')
    return replace_once(source, "dense_tile_parallel3_actual_operator_screen_completed",
                         "dense_tile_parallel2_long_k_actual_operator_screen_completed")


def main():
    parents = (ROOT / "build_dense_tile_parallel2_v1.py", ROOT / "run_dense_tile_parallel2_v1.py")
    outputs = (ROOT / "build_dense_tile_parallel2_long_k_v1.py", ROOT / "run_dense_tile_parallel2_long_k_v1.py")
    report = ROOT / "dense-tile-parallel2-long-k-v1-generation.json"
    if report.exists() or any(path.exists() for path in outputs):
        raise FileExistsError("Fresh long-K-only two-worker generation required")
    for parent, output, transform in zip(parents, outputs, (build_source, controller_source), strict=True):
        source = transform(parent.read_text())
        compile(source, str(output), "exec")
        with output.open("x") as stream:
            stream.write(source)
    with report.open("x") as stream:
        json.dump({"status": "generated_not_qualified",
                   "reason": "Broad two-worker route regressed wide K2560 shapes; only actual K6144/N2560 passed with ratio1.093109657 and126/128 improvements. Retain the identical shader only for that exact shape, keep all fallbacks original, and independently bracket again before full-native/model admission.",
                   "source_sha256": {str(path): sha(path) for path in
                                     (Path(__file__), ROOT / "dense_tile_parallel2_long_k_transform.py",
                                      ROOT / "dense_tile_parallel2_transform.py",
                                      ROOT / "dense_tile_parallel3_transform.py", *parents, *outputs)}},
                  stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()

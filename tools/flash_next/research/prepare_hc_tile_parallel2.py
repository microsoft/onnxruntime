import json
from pathlib import Path

from build_qmoe_node_replay import replace_once, sha
from prepare_hc_tile_parallel import build_source as four_build_source
from prepare_hc_tile_parallel import controller_source as four_controller_source


ROOT = Path(__file__).parent


def build_source(source):
    source = four_build_source(source)
    source = source.replace("TILE_PARALLEL4", "TILE_PARALLEL2").replace("TileParallel4", "TileParallel2")
    source = source.replace("tile-parallel4", "tile-parallel2")
    source = source.replace("hc_tile_parallel_transform", "hc_tile_parallel2_transform")
    source = source.replace("prepare_hc_tile_parallel.py", "prepare_hc_tile_parallel2.py")
    source = replace_once(source, '"physical_threads": 1024,', '"physical_threads": 512,')
    source = replace_once(source, '"independent_tile_workers": 4,', '"independent_tile_workers": 2,')
    return replace_once(source, "inputs = [Path(__file__),",
                        'inputs = [Path(__file__), Path(__file__).with_name("hc_tile_parallel_transform.py"),\n'
                        '              Path(__file__).with_name("prepare_hc_tile_parallel.py"),\n'
                        '              Path(__file__).with_name("prepare_hc_gate_staging.py"),')


def controller_source(source):
    source = four_controller_source(source)
    source = source.replace("TILE_PARALLEL4", "TILE_PARALLEL2").replace("TileParallel4", "TileParallel2")
    source = source.replace("tile-parallel4", "tile-parallel2").replace("tile_parallel4", "tile_parallel2")
    source = source.replace("prepare_hc_tile_parallel.py", "prepare_hc_tile_parallel2.py")
    return replace_once(source, '(":1024,1,1:" if selected else ":256,1,1:")',
                        '(":512,1,1:" if selected else ":256,1,1:")')


def main():
    parents = (ROOT / "build_hc_owned_pair34.py", ROOT / "run_hc_owned_pair34_v1.py")
    outputs = (ROOT / "build_hc_tile_parallel2_v1.py", ROOT / "run_hc_tile_parallel2_v1.py")
    report = ROOT / "hc-tile-parallel2-v1-generation.json"
    if report.exists() or any(path.exists() for path in outputs):
        raise FileExistsError("Fresh two-worker tile generation required")
    for parent, output, transform in zip(parents, outputs, (build_source, controller_source), strict=True):
        text = transform(parent.read_text())
        compile(text, str(output), "exec")
        with output.open("x") as stream:
            stream.write(text)
    with report.open("x") as stream:
        json.dump({
            "status": "generated_not_qualified",
            "source_sha256": {str(path): sha(path) for path in (
                Path(__file__), ROOT / "prepare_hc_tile_parallel.py", ROOT / "prepare_hc_gate_staging.py",
                ROOT / "hc_tile_parallel_transform.py", ROOT / "hc_tile_parallel2_transform.py",
                *parents, *outputs)},
        }, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()

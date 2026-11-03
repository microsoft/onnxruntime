import json
from pathlib import Path

from build_qmoe_node_replay import replace_once, sha
from prepare_hc_gate_staging import build_source as gate_build_source
from prepare_hc_gate_staging import controller_source as gate_controller_source


ROOT = Path(__file__).parent


def build_source(source):
    source = gate_build_source(source)
    source = source.replace("GATE_STAGING", "TILE_PARALLEL4").replace("GateStaging", "TileParallel4")
    source = source.replace("gate-staging", "tile-parallel4")
    source = source.replace("hc_gate_staging_transform", "hc_tile_parallel_transform")
    source = source.replace("prepare_hc_gate_staging.py", "prepare_hc_tile_parallel.py")
    source = replace_once(source, '"physical_threads": 256,', '"physical_threads": 1024,')
    source = replace_once(source, '"gates_computed_once_per_workgroup": True,',
                          '"independent_tile_workers": 4,\n'
                          '        "tile_scratch_full_word_f32_slots": 2560,\n'
                          '        "finite_f16_f32_f16_storage_roundtrip": True,')
    return replace_once(source, '"shared_storage_bytes": 27668,',
                        '"shared_storage_bytes": 37888,')


def controller_source(source):
    source = gate_controller_source(source)
    source = source.replace("GATE_STAGING", "TILE_PARALLEL4").replace("GateStaging", "TileParallel4")
    source = source.replace("gate-staging", "tile-parallel4").replace("gate_staging", "tile_parallel4")
    source = source.replace("prepare_hc_tile_parallel4.py", "prepare_hc_tile_parallel.py")
    return replace_once(
        source, 'or ":256,1,1:" not in event.get("args", {}).get("cache_key", "")',
        'or (":1024,1,1:" if selected else ":256,1,1:") not in event.get("args", {}).get("cache_key", "")')


def main():
    parents = (ROOT / "build_hc_owned_pair34.py", ROOT / "run_hc_owned_pair34_v1.py")
    outputs = (ROOT / "build_hc_tile_parallel4_v1.py", ROOT / "run_hc_tile_parallel4_v1.py")
    report = ROOT / "hc-tile-parallel4-v1-generation.json"
    if report.exists() or any(path.exists() for path in outputs):
        raise FileExistsError("Fresh tile-parallel generation required")
    for parent, output, transform in zip(parents, outputs, (build_source, controller_source), strict=True):
        text = transform(parent.read_text())
        compile(text, str(output), "exec")
        with output.open("x") as stream:
            stream.write(text)
    with report.open("x") as stream:
        json.dump({
            "status": "generated_not_qualified",
            "source_sha256": {str(path): sha(path) for path in
                              (Path(__file__), ROOT / "prepare_hc_gate_staging.py",
                               ROOT / "hc_tile_parallel_transform.py", *parents, *outputs)},
        }, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()

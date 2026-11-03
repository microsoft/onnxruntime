import json
from pathlib import Path

from build_qmoe_node_replay import replace_once, sha


ROOT = Path(__file__).parent


def builder_source(source):
    source = source.replace("parallel6", "parallel4")
    source = replace_once(source, '"physical_threads": 768, "independent_tile_workers": 6',
                          '"physical_threads": 512, "independent_tile_workers": 4')
    return replace_once(source,
                        '        ROOT / "dense_tile_parallel3_transform.py",',
                        '        ROOT / "dense_tile_parallel6_long_k_transform.py",\n'
                        '        ROOT / "dense_tile_parallel3_transform.py",')


def controller_source(source):
    source = source.replace("parallel6", "parallel4")
    source = replace_once(source, "threads = 768 if selected else 128", "threads = 512 if selected else 128")
    source = replace_once(source, 'manifest["physical_threads"] != 768', 'manifest["physical_threads"] != 512')
    source = replace_once(source, 'manifest["shared_storage_bytes"] != 27648',
                          'manifest["shared_storage_bytes"] != 19456')
    return replace_once(source, '"independent_tile_workers": 6', '"independent_tile_workers": 4')


def main():
    parent_builder = ROOT / "build_dense_tile_parallel6_long_k_ns_v1.py"
    parent_controller = ROOT / "run_dense_tile_parallel6_long_k_ns_v1.py"
    builder = ROOT / "build_dense_tile_parallel4_long_k_ns_v1.py"
    controller = ROOT / "run_dense_tile_parallel4_long_k_ns_v1.py"
    report = ROOT / "dense-tile-parallel4-long-k-ns-v1-generation.json"
    if any(path.exists() for path in (builder, controller, report)):
        raise FileExistsError("Fresh four-worker generation required")
    sources = {builder: builder_source(parent_builder.read_text()),
               controller: controller_source(parent_controller.read_text())}
    for path, source in sources.items():
        compile(source, str(path), "exec")
    for path, source in sources.items():
        with path.open("x") as stream:
            stream.write(source)
    with report.open("x") as stream:
        json.dump({"status": "generated_not_qualified", "workers": 4, "waves": 2,
                   "tail_workers_do_not_write_out_of_bounds": True,
                   "source_sha256": {str(path): sha(path) for path in
                                     (Path(__file__), ROOT / "dense_tile_parallel4_long_k_transform.py",
                                      parent_builder, parent_controller, builder, controller)}},
                  stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()

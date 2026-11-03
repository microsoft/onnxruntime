from pathlib import Path

from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent


if __name__ == "__main__":
    source = (ROOT / "run_dense_paired_matrix_batch8_probe.py").read_text()
    source = source.replace("batch8", "batch4").replace("Batch8", "Batch4")
    source = replace_once(source,
        '    build_path = BUILD / "build-manifest.json"\n',
        '''    limits_path = ROOT / "native-matrix-limit-probe-v1-outcome.json"
    limits = json.loads(limits_path.read_text())
    for path, digest in limits["source_and_evidence_sha256"].items():
        if sha(Path(path)) != digest:
            raise RuntimeError(f"Changed device-limit proof: {path}")
    if (limits["status"] != "actual_device_limits_observed"
            or not limits["paired_matrix_configuration"]
            or limits["workgroup_storage_bytes"] < 46080
            or limits["workgroup_size_x"] < 512
            or limits["workgroup_invocations"] < 512):
        raise RuntimeError("Observed device limits do not admit the new four-slot design")
    build_path = BUILD / "build-manifest.json"
''')
    source = replace_once(source, "                prerequisite, build_path, ORIGINAL, provider, binary]",
                          "                prerequisite, limits_path, build_path, ORIGINAL, provider, binary]")
    output = ROOT / "run_dense_paired_matrix_batch4_probe.py"
    with output.open("x") as stream:
        stream.write(source)
    print(output.name)

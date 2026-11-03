import json
from pathlib import Path

from build_qmoe_node_replay import replace_once, sha
import prepare_int8_lm_head_full_a_native as original
from prepare_int8_lm_head_full_a_native import MODEL, PROOF, region_sha


ROOT = Path(__file__).parent
DATA = original.REPO / "fixtures-v3"
RUNNER = ROOT / "run_int8_lm_head_full_a_native_v3.py"


def runner_source():
    source = (ROOT / "run_int8_lm_head_full_a_native.py").read_text()
    source = replace_once(source,
        "from prepare_int8_lm_head_full_a_native import DATA, MODEL, PROOF, region_sha",
        "from prepare_int8_lm_head_full_a_native_v3 import DATA, MODEL, PROOF, region_sha")
    source = replace_once(source, 'REFERENCES = PROOF / "original-references"',
                          'REFERENCES = PROOF / "original-references-v3"')
    source = replace_once(source, '    assert (DATA / "model.onnx.data").samefile(MODEL / "model.onnx.data")',
                          '    assert not (DATA / "model.onnx.data").exists()\n'
                          '    assert (DATA / "actual-head.data").is_file()')
    source = replace_once(source,
        '            assert region_sha(MODEL / external["location"], int(external["offset"]), int(external["length"])) == tensor["payload_sha256"]',
        '            assert region_sha(MODEL / external["location"], int(external["offset"]), int(external["length"])) == tensor["payload_sha256"]\n'
        '            local = tensor["fixture_external"]\n'
        '            assert region_sha(DATA / local["location"], int(local["offset"]), int(local["length"])) == tensor["payload_sha256"]')
    source = replace_once(source, '    evidence.extend([fixture_path, MODEL / "model.onnx", MODEL / "genai_config.json",',
                          '    evidence.extend([fixture_path, DATA / "actual-head.data", MODEL / "model.onnx", MODEL / "genai_config.json",')
    return source.replace("int8-lm-head-full-a-v1-native", "int8-lm-head-full-a-v3-native")


if __name__ == "__main__":
    import onnx
    if RUNNER.exists():
        raise FileExistsError(RUNNER)
    original.DATA = DATA
    original.prepare_fixtures()
    link = DATA / "model.onnx.data"
    assert link.is_symlink() and link.samefile(MODEL / "model.onnx.data")
    manifest_path = DATA / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    row, = [record for record in manifest["fixtures"] if record["actual_weights"]]
    path = Path(row["path"])
    model = onnx.load(path, load_external_data=False)
    local_file = DATA / "actual-head.data"
    offset = 0
    with local_file.open("xb") as destination:
        for tensor, record in zip(model.graph.initializer, row["tensors"]):
            external = record["external"]
            remaining = int(external["length"])
            with (MODEL / external["location"]).open("rb") as source:
                source.seek(int(external["offset"]))
                while remaining:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise RuntimeError("Truncated actual tensor source")
                    destination.write(chunk)
                    remaining -= len(chunk)
            local = dict(location=local_file.name, offset=str(offset), length=external["length"])
            record["fixture_external"] = local
            for item in tensor.external_data:
                if item.key in local:
                    item.value = local[item.key]
            offset += int(external["length"])
    link.unlink()
    onnx.save(model, path)
    onnx.checker.check_model(str(path))
    row["sha256"] = sha(path)
    manifest["external_weights_symlink_to_frozen_model"] = False
    manifest["copied_actual_tensor_bytes"] = offset
    for record in row["tensors"]:
        local = record["fixture_external"]
        assert region_sha(local_file, int(local["offset"]), int(local["length"])) == record["payload_sha256"]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    with RUNNER.open("x") as stream:
        stream.write(runner_source())
    print(RUNNER)

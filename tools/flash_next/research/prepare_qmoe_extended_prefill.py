import json
from pathlib import Path
import subprocess

import onnx

from build_qmoe_node_replay import sha


DIRECTORY = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-extended-prefill/fixtures-v1")


def main():
    original = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-node-replay")
    fixtures = [f for f in json.loads((original / "fixtures.json").read_text()) if f["rows"] == 16]
    assert len(fixtures) == 6 and all(f["hidden"] == 2560 for f in fixtures)
    DIRECTORY.mkdir(parents=True, exist_ok=False)
    data = original / "fixtures/model.onnx.data"
    subprocess.run(["sudo", "-n", "ln", str(data), str(DIRECTORY / data.name)], check=True)
    assert data.samefile(DIRECTORY / data.name)
    sources = {}
    for fixture in fixtures:
        source = original / "fixtures" / fixture["fixture"]
        model = onnx.load(source, load_external_data=False)
        assert {v.name for v in model.graph.input} == {"x", "router"}
        for value in (*model.graph.input, *model.graph.output):
            dims = value.type.tensor_type.shape.dim
            assert len(dims) == 2 and dims[0].dim_value == 16
            dims[0].ClearField("dim_value")
            dims[0].dim_param = "tokens"
        name = f"expert_layer{fixture['layer']}_dynamic.onnx"
        onnx.save(model, DIRECTORY / name)
        sources[name] = dict(source=str(source), source_sha256=sha(source),
                             fixture_sha256=sha(DIRECTORY / name))
    (DIRECTORY / "manifest.json").write_text(json.dumps(dict(
        fixtures=sources, weights_hardlinked=True, source_weights=str(data),
        weights_bytes=data.stat().st_size), indent=2) + "\n")
    print(f"Prepared {len(fixtures)} dynamic original-layer fixtures; weights hardlinked, not copied.")


if __name__ == "__main__":
    main()

import json
from pathlib import Path

import onnx

from build_qmoe_node_replay import sha
from test_gdn_prep_fusion import fixture


OUTPUT = Path("/workspace/qwen38-int4-gdn-prep-node-replay/fixtures-v1")


def main(output=OUTPUT):
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for batch, length in ((1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 16), (2, 1)):
        model = fixture(batch, length)
        onnx.checker.check_model(model)
        path = output / f"gdn_prep_b{batch}_l{length}.onnx"
        onnx.save_model(model, path)
        count = batch * (length * (10240 + 6144 + 96) + 30720)
        rows.append(dict(path=path.name, batch=batch, length=length,
                         output_elements=count, sha256=sha(path),
                         single_row_replay=batch == length == 1))
    (output / "manifest.json").write_text(json.dumps(dict(
        status="gdn_prep_changing_gpu_input_fixtures_prepared", inputs_each=5, outputs_each=5,
        fixtures=rows, rounds=128, pending_windows=[1, 64],
        total_comparisons=sum(row["output_elements"] for row in rows) * 128 * 2), indent=2) + "\n")
    print(json.dumps(dict(fixtures=len(rows), total_comparisons=sum(
        row["output_elements"] for row in rows) * 128 * 2)))


if __name__ == "__main__":
    main()

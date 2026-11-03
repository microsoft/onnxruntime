import json
from pathlib import Path

import onnx

from build_qmoe_node_replay import sha
from prepare_early_scratch_batch import OUTPUT as PREVIOUS, SOURCE, batch_model


OUTPUT = PREVIOUS.parent / "native-v2"
FIXTURE = SOURCE.with_name("early_scratch_batch32_v2.onnx")


def main():
    if FIXTURE.exists():
        raise FileExistsError(FIXTURE)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    original = onnx.load(SOURCE, load_external_data=False)
    onnx.save(batch_model(original), FIXTURE)
    onnx.checker.check_model(str(FIXTURE))
    (OUTPUT / "generation.json").write_text(json.dumps({
        "status": "generated_with_external_weights_inside_model_directory",
        "qmoe_nodes": 32,
        "concat_nodes": 1,
        "output_elements": 81920,
        "weight_bytes_copied": 0,
        "new_weight_aliases": 0,
        "fixture": str(FIXTURE),
        "benchmark": str(PREVIOUS / "native_benchmark"),
        "same_weight_inputs_and_routes_per_batch": True,
        "readback_overhead_amortized_not_model_latency": True,
        "source_sha256": {str(path): sha(path) for path in
                          (Path(__file__), Path(__file__).with_name("prepare_early_scratch_batch.py"),
                           SOURCE, FIXTURE, PREVIOUS / "native_benchmark.cpp", PREVIOUS / "native_benchmark")},
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()

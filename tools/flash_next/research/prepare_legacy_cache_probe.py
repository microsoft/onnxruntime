import argparse
import copy
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
source = onnx.load(args.source / "model.onnx", load_external_data=False)
config = json.loads((args.source / "genai_config.json").read_text())
input_names = {value.name for value in source.graph.input}
nodes = []
for output in source.graph.output:
    if output.name == "logits":
        continue
    suffix = output.name.removeprefix("present.")
    past = ("past_key_values." if suffix.endswith((".key", ".value")) else "past.") + suffix
    if past not in input_names:
        raise ValueError(f"No matching cache input for {output.name}: {past}")
    nodes.append(helper.make_node("Identity", [past], [output.name]))
logits = numpy_helper.from_array(
    np.zeros((1, 1, config["model"]["vocab_size"]), dtype=np.float32), "logits",
)
graph = helper.make_graph(nodes, "legacy_cache_contract_probe", source.graph.input,
                          source.graph.output, [logits])
probe = helper.make_model(graph, opset_imports=source.opset_import)
probe.ir_version = source.ir_version
onnx.checker.check_model(probe)
for name, sharing in (("growing", False), ("shared", True), ("fixed_missing_length", True)):
    destination = args.output / name
    destination.mkdir(parents=True, exist_ok=True)
    variant = copy.deepcopy(config)
    variant["model"]["decoder"]["session_options"] = {
        "provider_options": [], "intra_op_num_threads": 1, "inter_op_num_threads": 1,
    }
    variant["search"].update(past_present_share_buffer=sharing, max_length=40)
    variant_graph = copy.deepcopy(probe)
    if name == "fixed_missing_length":
        for value in list(variant_graph.graph.input) + list(variant_graph.graph.output):
            if value.name.endswith(".indexer_key"):
                dimension = value.type.tensor_type.shape.dim[1]
                dimension.ClearField("dim_param")
                dimension.dim_value = 40
    onnx.save(variant_graph, destination / "model.onnx")
    (destination / "genai_config.json").write_text(json.dumps(variant, indent=2) + "\n")
print(args.output)

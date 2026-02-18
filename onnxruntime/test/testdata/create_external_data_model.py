# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import struct

from onnx import TensorProto, helper, save


def create_model(output_path, external_data_rel_path):
    inputs = []
    nodes = []
    tensors = []
    outputs = []

    # Create input tensor info
    input_ = helper.make_tensor_value_info("input", TensorProto.FLOAT, [100])
    inputs.append(input_)

    # Create tensor with external data
    # The data is just a sequence of 100 floats
    vals = [float(i) for i in range(100)]
    tensor = helper.make_tensor(name="external_weights", data_type=TensorProto.FLOAT, dims=[100], vals=vals)
    tensors.append(tensor)

    # Set external data location
    tensor.data_location = TensorProto.EXTERNAL

    # Check if external_data_rel_path is valid
    if not external_data_rel_path:
        raise ValueError("external_data_rel_path cannot be empty")

    # Location entry
    entry1 = tensor.external_data.add()
    entry1.key = "location"
    entry1.value = external_data_rel_path

    # Offset entry
    entry2 = tensor.external_data.add()
    entry2.key = "offset"
    entry2.value = "0"

    # Length entry
    entry3 = tensor.external_data.add()
    entry3.key = "length"
    entry3.value = str(len(vals) * 4)  # 4 bytes per float

    # Create constant node using the tensor
    nodes.append(helper.make_node(op_type="Constant", inputs=[], outputs=["const_output"], value=tensor))

    # Create Add node to use input and const_output
    nodes.append(helper.make_node(op_type="Add", inputs=["input", "const_output"], outputs=["output"]))

    # Create output tensor info
    outputs.append(helper.make_tensor_value_info("output", TensorProto.FLOAT, [100]))

    # Build the graph
    graph = helper.make_graph(nodes, "test_whitelist", inputs, outputs, tensors)

    # Create the model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    # Save the model
    # We need to manually write the external data file if we want it to exist
    # But onnx.save will try to write it relative to the model path if we don't specify otherwise
    # However, for the test we want to rely on ONNX runtime's loading behavior.
    # The external data file needs to exist for the load to succeed (after whitelist check).

    model_dir = os.path.dirname(output_path)
    if not model_dir:
        model_dir = "."

    external_data_full_path = os.path.join(model_dir, external_data_rel_path)
    external_data_dir = os.path.dirname(external_data_full_path)

    if external_data_dir and not os.path.exists(external_data_dir):
        os.makedirs(external_data_dir)

    # Create the external data file with raw bytes
    with open(external_data_full_path, "wb") as f:
        f.writelines(struct.pack("f", v) for v in vals)

    # Save the model, but we've already written the external data.
    # Validating the model might fail if we don't handle paths carefully during save.
    # Actually, let's just use onnx.save, it handles external data writing if we provide location.
    # Wait, we manually set external data location in the tensor proto.
    # onnx.save doesn't automatically move data to that location unless we use save_model with external_data=True etc.
    # But here we constructed the proto manually with EXTERNAL location.
    # So onnx.save will just save the proto. It won't write the external file because the data IS in vals (raw_data is undefined/empty in proto if we use vals, but logic is complex).
    # actually helper.make_tensor with vals puts data in the specific type field (float_data).
    # If we want it to be external, we should clear float_data and set data_location.

    tensor.ClearField("float_data")

    # Now save the model proto
    save(model, output_path)
    print(f"Model saved to {output_path}")
    print(f"External data file created at {external_data_full_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Path to save the ONNX model")
    parser.add_argument("--external_data", required=True, help="Relative path for external data")
    args = parser.parse_args()

    create_model(args.output, args.external_data)

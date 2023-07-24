import argparse

import onnx

parser = argparse.ArgumentParser(description="ONNX file analyzer for performance investigation.")
parser.add_argument("onnx_file", type=str, help="ONNX file to analyze")
args = parser.parse_args()


def process_file(onnx_file):
    model = onnx.load(onnx_file)

    # Map from output arg to the producer of the output.
    output_to_node = {}
    for node in model.graph.node:
        for o in node.output:
            output_to_node[o] = node

    aten_ops = []
    python_ops = []
    memcpu_ops = []
    cast_ops = []
    msgs = []

    for node in model.graph.node:
        if "Memcpy" in node.op_type:
            memcpu_ops.append(f"{node.op_type} {node.name}")
        if node.op_type == "Cast":
            cast_ops.append(f"{node.name}")
        if node.op_type == "ATen":
            for attr in node.attribute:
                if attr.name == "operator":
                    aten_ops.append(f"{node.name}: {attr.s.decode('utf-8')}")  # noqa: PERF401
        if node.op_type == "PythonOp":
            for attr in node.attribute:
                if attr.name == "name":
                    python_ops.append(f"{node.name}: {attr.s.decode('utf-8')}")  # noqa: PERF401

        # Look for stand-alone Dropout node in *_execution_model_<mode>.onnx graph.
        # Examine whether it should be fused with surrounding Add ops into BiasDropout node.
        if node.op_type == "Dropout" and len(node.input) == 1:
            prev = output_to_node[node.input[0]]
            if prev.op_type == "Add":
                msgs.append(
                    f"Examine whether {node.name} should be fused with the leading {prev.name} op into BiasDropout node."
                )

        # Look for stand-alone Softmax node in *_execution_model_<mode>.onnx graph.
        # Examine whether it should be fused with the leading Add ops into BiasSoftmax node.
        if node.op_type == "Softmax" and len(node.input) == 1:
            prev = output_to_node[node.input[0]]
            if prev.op_type == "Add":
                msgs.append(
                    f"Examine whether {node.name} should be fused with the leading {prev.name} op into BiasSoftmax node."
                )

    if aten_ops:
        print("ATen op found:")
        for line in aten_ops:
            print(line)
        print(10 * "-")

    if python_ops:
        print("PythonOp found:")
        for line in python_ops:
            print(line)
        print(10 * "-")

    if memcpu_ops:
        print("Memcpu ops found:")
        for line in memcpu_ops:
            print(line)
        print(10 * "-")

    if cast_ops:
        print("Cast ops found:")
        for line in cast_ops:
            print(line)
        print(10 * "-")

    for line in msgs:
        print(line)


def main():
    process_file(args.onnx_file)


if __name__ == "__main__":
    main()

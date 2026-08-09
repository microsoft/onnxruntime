import argparse
import os

import onnx


def export_and_recurse(node, attribute, output_dir, level):
    name = node.name
    name = name.replace("/", "_")
    sub_model = onnx.ModelProto()
    sub_model.graph.MergeFrom(attribute.g)
    filename = "L" + str(level) + "_" + node.op_type + "_" + attribute.name + "_" + name + ".onnx"
    onnx.save_model(sub_model, os.path.join(output_dir, filename))
    dump_subgraph(sub_model, output_dir, level + 1)


def dump_subgraph(model, output_dir, level=0):
    graph = model.graph

    for node in graph.node:
        if node.op_type == "Scan" or node.op_type == "Loop":
            body_attribute = next(iter(filter(lambda attr: attr.name == "body", node.attribute)))
            export_and_recurse(node, body_attribute, output_dir, level)
        if node.op_type == "If":
            then_attribute = next(iter(filter(lambda attr: attr.name == "then_branch", node.attribute)))
            else_attribute = next(iter(filter(lambda attr: attr.name == "else_branch", node.attribute)))
            export_and_recurse(node, then_attribute, output_dir, level)
            export_and_recurse(node, else_attribute, output_dir, level)


def parse_args():
    parser = argparse.ArgumentParser(
        os.path.basename(__file__), description="Dump all subgraphs from an ONNX model into separate onnx files."
    )
    parser.add_argument("-m", "--model", required=True, help="model file")
    parser.add_argument("-o", "--out", required=True, help="output directory")
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = args.model
    out = os.path.abspath(args.out)

    if not os.path.exists(out):
        os.makedirs(out)

    model = onnx.load_model(model_path)
    dump_subgraph(model, out)


if __name__ == "__main__":
    main()

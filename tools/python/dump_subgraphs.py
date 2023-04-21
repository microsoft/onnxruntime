import argparse
import os

import onnx


def export_and_recurse(node, attribute, attribute_has_single_graph, can_check_attribute_type, output_dir, level):
    node_name = node.name.replace("/", "_")

    attr_graph_protos = [attribute.g] if attribute_has_single_graph else attribute.graphs

    for idx, attr_graph_proto in enumerate(attr_graph_protos):
        sub_model = onnx.ModelProto()
        sub_model.graph.MergeFrom(attr_graph_proto)
        attribute_graph_id = attribute.name if attribute_has_single_graph else f"{attribute.name}_{idx}"
        filename = f"L{level}_{node.op_type}_{attribute_graph_id}_{node_name}.onnx"
        onnx.save_model(sub_model, os.path.join(output_dir, filename))
        dump_subgraph(sub_model, can_check_attribute_type, output_dir, level + 1)


def dump_subgraph(model, can_check_attribute_type, output_dir, level=0):
    graph = model.graph

    for node in graph.node:
        if can_check_attribute_type:
            graph_attributes = [
                attribute
                for attribute in node.attribute
                if attribute.type in [onnx.AttributeProto.AttributeType.GRAPH, onnx.AttributeProto.AttributeType.GRAPHS]
            ]
            for graph_attribute in graph_attributes:
                attribute_has_single_graph = graph_attribute.type == onnx.AttributeProto.AttributeType.GRAPH
                export_and_recurse(
                    node, graph_attribute, attribute_has_single_graph, can_check_attribute_type, output_dir, level
                )

        else:
            # can't check AttributeProto.type, so check for known ONNX ops with graph attributes
            if node.op_type == "Scan" or node.op_type == "Loop":
                attribute_has_single_graph = True
                body_attribute = list(filter(lambda attr: attr.name == "body", node.attribute))[0]
                export_and_recurse(
                    node, body_attribute, attribute_has_single_graph, can_check_attribute_type, output_dir, level
                )
            if node.op_type == "If":
                attribute_has_single_graph = True
                then_attribute = list(filter(lambda attr: attr.name == "then_branch", node.attribute))[0]
                else_attribute = list(filter(lambda attr: attr.name == "else_branch", node.attribute))[0]
                export_and_recurse(
                    node, then_attribute, attribute_has_single_graph, can_check_attribute_type, output_dir, level
                )
                export_and_recurse(
                    node, else_attribute, attribute_has_single_graph, can_check_attribute_type, output_dir, level
                )


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

    # AttributeProto.type is available starting from IR_VERSION_2017_10_30
    can_check_attribute_type = model.ir_version >= onnx.Version.IR_VERSION_2017_10_30

    dump_subgraph(model, can_check_attribute_type, out)


if __name__ == "__main__":
    main()

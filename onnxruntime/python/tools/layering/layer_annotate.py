# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import logging
import pathlib

import onnx


def get_logger(name, level=logging.DEBUG):
    logging.basicConfig(format="%(asctime)s %(name)s [%(levelname)s] - %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


def getargs():
    argparser = argparse.ArgumentParser(
        description="Read a config file with a list of node annotations and apply them to an ONNX model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argparser.add_argument(
        "--config_file_path",
        type=pathlib.Path,
        required=True,
        help="Path to the configuration file with node annotations.",
    )
    argparser.add_argument(
        "--model_path",
        type=pathlib.Path,
        required=True,
        help="Path to a single model to process.",
    )
    argparser.add_argument(
        "--annotated_model",
        type=pathlib.Path,
        required=True,
        help="Path to write the annotated model to.",
    )

    return argparser.parse_args()


def read_annotation_config(config_file_path):
    """
    Reads a configuration file to map substrings to annotations.

    The file format is expected to be:
    annotation_string: substring1, substring2, ...

    The same annotation string can appear multiple times.
    The node names in the configuration are treated as substrings.

    Args:
        config_file_path (str or Path): Path to the configuration file.

    Returns:
        list: A list of tuples (substring, annotation_string).
    """
    substring_annotations = []
    with open(config_file_path) as f:
        for unstripped_line in f:
            line = unstripped_line.strip()
            if not line:
                continue
            parts = line.split(":", 1)
            if len(parts) < 2:
                continue
            annotation = parts[0].strip()
            substrings = parts[1].split(",")
            for substr in substrings:
                substring = substr.strip()
                if substring:
                    substring_annotations.append((substring, annotation))
    return substring_annotations


def process_nodes(nodes, substring_annotations):
    """
    Helper function to process a list of nodes sequentially.
    """
    logger = get_logger("annotate_model")
    logger.info(f"Processing {len(nodes)} nodes.")

    for node in nodes:
        matched_annotation = None
        for substring, annotation in substring_annotations:
            if substring in node.name:
                matched_annotation = annotation

        if matched_annotation:
            # Check if annotation already exists
            entry = None
            for prop in node.metadata_props:
                if prop.key == "layer_ann":
                    entry = prop
                    break

            if entry:
                entry.value = matched_annotation
            else:
                entry = node.metadata_props.add()
                entry.key = "layer_ann"
                entry.value = matched_annotation

        # Recurse into subgraphs for control flow nodes
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                annotate_graph(attr.g, substring_annotations)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for sub_graph in attr.graphs:
                    annotate_graph(sub_graph, substring_annotations)


def annotate_graph(graph, substring_annotations):
    """
    Recursively applies annotations to nodes where a configured substring appears in the node name.

    This function iterates over all nodes in the given graph. It checks if any
    substring from the configuration appears in the node's name. If matched,
    it adds or updates a metadata property with key 'layer_ann' containing
    the annotation string. If multiple substrings match, the last one defined
    in the configuration list applies.

    It also handles control flow nodes (like 'If' or 'Loop') by recursively
    processing their subgraphs (attributes of type GRAPH or GRAPHS).

    Args:
        graph (onnx.GraphProto): The ONNX graph to process.
        substring_annotations (list): A list of tuples (substring, annotation_string).
    """
    process_nodes(graph.node, substring_annotations)


def annotate_model(model, substring_annotations):
    """
    Annotates an ONNX model with metadata based on a provided mapping.

    This function serves as the entry point to annotate the model's graph.
    It delegates the work to `annotate_graph`, which recursively processes
    all nodes in the main graph and any nested subgraphs.

    Args:
        model (onnx.ModelProto): The ONNX model to annotate.
        substring_annotations (list): A list of tuples (substring, annotation_string).
    """
    annotate_graph(model.graph, substring_annotations)


if __name__ == "__main__":
    args = getargs()
    logger = get_logger("annotate_model")

    # Read the mapping from the configuration file
    substring_annotations = read_annotation_config(args.config_file_path)

    logger.info(f"Loading model from {args.model_path}")
    onnx_model = onnx.load(args.model_path, load_external_data=False)

    logger.info(f"Applying annotations from {args.config_file_path}")
    annotate_model(onnx_model, substring_annotations)

    logger.info(f"Saving annotated model to {args.annotated_model}")
    onnx.save_model(onnx_model, args.annotated_model)

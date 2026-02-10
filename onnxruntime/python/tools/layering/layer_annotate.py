import argparse
import concurrent.futures
import logging
import os
import pathlib
import threading

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
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(":", 1)
            if len(parts) < 2:
                continue
            annotation = parts[0].strip()
            substrings = parts[1].split(",")
            for substring in substrings:
                substring = substring.strip()
                if substring:
                    substring_annotations.append((substring, annotation))
    return substring_annotations


def process_nodes(nodes, substring_annotations):
    """
    Helper function to process a list of nodes sequentially.
    """
    logger = get_logger("annotate_model")
    logger.info(f"Thread {threading.get_ident()} processing {len(nodes)} nodes.")

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
                annotate_graph(attr.g, substring_annotations, parallel=False)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for sub_graph in attr.graphs:
                    annotate_graph(sub_graph, substring_annotations, parallel=False)


def annotate_graph(graph, substring_annotations, parallel=False):
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
        parallel (bool): If True, process the graph's nodes in parallel chunks.
    """
    if parallel:
        logger = get_logger("annotate_model")
        num_cores = os.cpu_count() or 1
        nodes = graph.node
        total_nodes = len(nodes)
        min_nodes_per_thread = 1000

        if total_nodes > 0:
            # Ensure each thread processes at least min_nodes_per_thread, if possible
            max_workers = max(1, total_nodes // min_nodes_per_thread)
            num_workers = min(num_cores, max_workers)

            logger.info(
                f"Parallel processing configuration: Total Nodes={total_nodes}, Cores={num_cores}. "
                f"Calculated Workers={num_workers} (Min nodes per thread={min_nodes_per_thread})."
            )

            chunks = []
            start_index = 0
            base_chunk_size = total_nodes // num_workers
            remainder = total_nodes % num_workers

            for i in range(num_workers):
                # Distribute the remainder (extra nodes) across the first 'remainder' threads
                # To avoid the last worker processing very small amount of nodes
                current_chunk_size = base_chunk_size + (1 if i < remainder else 0)
                end_index = start_index + current_chunk_size
                chunks.append(nodes[start_index:end_index])
                start_index = end_index

            # Use current thread for one of the chunks to avoid idle main thread
            if num_workers > 1:
                # Execute num_workers - 1 chunks in background threads
                # Execute the last chunk in the current (main) thread
                background_chunks = chunks[:-1]
                main_chunk = chunks[-1]

                logger.info(f"Dispatching {len(background_chunks)} chunks to thread pool and 1 chunk to main thread.")

                with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers - 1) as executor:
                    futures = [
                        executor.submit(process_nodes, chunk, substring_annotations) for chunk in background_chunks
                    ]

                    # Run last chunk here
                    process_nodes(main_chunk, substring_annotations)

                    concurrent.futures.wait(futures)
            else:
                # Only 1 worker needed, run in current thread
                logger.info("Using single thread (current) for processing.")
                process_nodes(chunks[0], substring_annotations)
    else:
        process_nodes(graph.node, substring_annotations)


def annotate_model(model, substring_annotations):
    """
    Annotates an ONNX model with metadata based on a provided mapping.

    This function serves as the entry point to annotate the model's graph.
    It delegates the work to `annotate_graph` enabling parallel processing for the main graph.

    Args:
        model (onnx.ModelProto): The ONNX model to annotate.
        substring_annotations (list): A list of tuples (substring, annotation_string).
    """
    annotate_graph(model.graph, substring_annotations, parallel=True)


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

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import onnx
import onnxruntime as ort
import os
import pathlib


def create_initializer_map(graph, initializers):
    '''
    For each initializer create a map between where it's defined and where it's used.
    :param graph: Current graph or subgraph.
    :param initializers: Initializer map to update.
    '''

    for i in graph.initializer:
        # if not seen before create empty entry.
        # The entry is a mapping from the initializer name to a map of {graph defined in: set(graph used in)}
        # as an initializer could be redefined in a subgraph and shadow the initializer in the ancestor graph.
        if i.name not in initializers:
            initializers[i.name] = {}

        # add entry for this graph with empty set for the graph/s it is used in.
        # the set is updated in track_initializer_usage.
        initializers[i.name][graph] = set()

    for node in graph.node:
        # recurse into subgraph for control flow nodes (Scan/Loop/If)
        for attr in node.attribute:
            if attr.HasField('g'):
                create_initializer_map(attr.g, initializers)


def track_initializer_usage(graph, initializers, ancestors):
    '''
    Update usage for each initializer. We track the graph instance/s each initializer is used in.
    :param graph: Current graph or subgraph.
    :param initializers: Map of initializer name to per-graph usage which is updated during processing.
    :param ancestors: Ordered ancestor list to match correct graph if the initializer is defined in multiple.
    '''

    for node in graph.node:
        for i in node.input:
            if i in initializers:
                initializer_entry = initializers[i]
                if graph in initializer_entry:
                    # initializer is defined and used in this graph
                    initializer_entry[graph].add(graph)
                else:
                    # initializer is defined in an ancestor graph and used in this graph.
                    # reverse iterate ancestors to find first graph where initializer was defined.
                    # this value shadows any others.
                    for g in reversed(ancestors):
                        if g in initializer_entry:
                            initializer_entry[g].add(graph)
                            break

        # recurse into subgraph for control flow nodes (Scan/Loop/If)
        for attr in node.attribute:
            if attr.HasField('g'):
                # add current graph to end of ancestors while recursing.
                ancestors.append(graph)
                track_initializer_usage(attr.g, initializers, ancestors)
                ancestors.pop(-1)


def move_initializers(initializers):
    '''
    If an initializer is only used in one graph, and it is not defined in that graph, move it there.
    :param initializers: Map with initializer usage.
    '''

    for name, entry in initializers.items():
        for graph_defined_in, graphs_used_in in entry.items():
            if len(graphs_used_in) == 1:
                graph_used_in = list(graphs_used_in)[0]
                if graph_defined_in != graph_used_in:
                    print(f"Moving {name} from "
                          f"graph with name={graph_defined_in.name}, docstring={graph_defined_in.doc_string} to "
                          f"graph with name={graph_used_in.name}, docstring={graph_used_in.doc_string}")
                    for idx in range(len(graph_defined_in.initializer)):
                        i = graph_defined_in.initializer[idx]
                        if i.name == name:
                            graph_used_in.initializer.append(i)
                            del graph_defined_in.initializer[idx]
                            break


def parse_args():
    parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        description='''Move initializers to a lower subgraph if possible to reduce the outer scope values used in the
        model. This should reduce the cost of handling them during graph resolution if the model has a large number
        of this sort of initializers.
        '''
    )

    parser.add_argument('--optimization_level', default='disable',
                        choices=['disable', 'basic'],
                        help="Level to optimize ONNX model with when converting any Constant nodes to initializers "
                             "during pre-processing."
                        )

    parser.add_argument('model', type=pathlib.Path, help='Path to ONNX model to update.')

    return parser.parse_args()


def move_initializers_down():
    args = parse_args()
    model_path = args.model.resolve()

    if not model_path.is_file():
        raise FileNotFoundError("Model path '{}' is not a file.".format(model_path))

    # Pre-process the model using ORT to convert any Constant nodes to initializers.
    # Do any additional 'basic' level optimizations if requested (e.g. constant folding).
    #
    # For each initializer, build a map from the GraphProto instance it's defined in to the GraphProto instance/s
    # it's used in.
    #   - also handle an initializer in a subgraph shadowing one in an ancestor graph when they both have the same name.
    #
    # For each initializer that is used in one graph, and that graph is different to where the initializer is defined,
    # move it.

    tmp_model_path = str(model_path.with_suffix('.tmp' + model_path.suffix))
    final_model_path = str(model_path.with_suffix('.updated' + model_path.suffix))

    # hack in a hash function so that we can use the GraphProto as a key. we only have one instance of each GraphProto
    # in the model so the address of that is fine as it will be unique and won't change.
    onnx.GraphProto.__hash__ = lambda self: id(self)

    # pre-process model in ORT to convert Constant nodes to initializers and perform any requested optimizations
    # if we haven't done that already.
    if not os.path.isfile(tmp_model_path):
        so = ort.SessionOptions()
        so.optimized_model_filepath = str(tmp_model_path)
        so.graph_optimization_level = \
            ort.GraphOptimizationLevel.ORT_ENABLE_BASIC if args.optimization_level == 'basic' \
            else ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        # we just need to create the session to do the pre-processing and write to tmp_model_path
        _ = ort.InferenceSession(str(model_path), so)

    model = onnx.load(tmp_model_path)

    # the model is now loaded into memory so we can remove the temporary file
    # NOTE: can comment out this line if you're testing and want to speed things up as the pre-processing
    #       is deterministic and can take a while for a large model.
    os.remove(tmp_model_path)

    # map {initializer name: {graph_defined_in: set(graphs used in)}}
    initializer_map = {}
    create_initializer_map(model.graph, initializer_map)
    track_initializer_usage(model.graph, initializer_map, [])
    move_initializers(initializer_map)

    onnx.checker.check_model(model)
    onnx.save(model, final_model_path)


if __name__ == '__main__':
    move_initializers_down()

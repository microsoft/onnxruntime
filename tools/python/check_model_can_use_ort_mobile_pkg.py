# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Helper script that will check if the types and operators used in an ONNX model
# would be supported by the pre-built ORT Mobile package.

import argparse
import onnx
import os
import pathlib
import sys
from onnx import shape_inference
from util import reduced_build_config_parser

cpp_to_tensorproto_type = {
    'float': 1,
    'uint8_t': 2,
    'int8_t': 3,
    'uint16_t': 4,
    'int16_t': 5,
    'int32_t': 6,
    'int64_t': 7,
    'std::string': 8,
    'bool': 9,
    'MLFloat16': 10,
    'double': 11,
    'uint32_t': 12,
    'uint64_t': 13,
    'Complex64': 14,   # not supported by ORT
    'Complex128': 15,  # not supported by ORT
    'BFloat16': 16
}

tensorproto_type_to_cpp = {v: k for k, v in cpp_to_tensorproto_type.items()}


def check_graph(graph, opsets, required_ops, global_types, special_types, unsupported_ops):
    '''
    Check the graph and any subgraphs for usage of types or operators which we know are not supported.
    :param graph: Graph to process.
    :param opsets: Map of domain to opset version that the model imports.
    :param required_ops: Operators that are included in the pre-built package.
    :param global_types: Types globally enabled in the pre-built package.
    :param special_types: Types that are always enabled for a subset of operators and are _usually_ supported but are
                          are not guaranteed to be. We would need to add a lot of infrastructure to know for sure so
                          currently we treat them as supported.
    :param unsupported_ops: Set of unsupported operators that is updated as they are found and returned to the caller.
    :return: Returns whether the graph uses unsupported operators or types.
    '''
    has_unsupported_types = False
    value_info_map = {vi.name: vi for vi in graph.value_info}

    def _is_type_supported(value_info, description):
        is_supported = True
        type_name = i.type.WhichOneof('value')
        if type_name == 'tensor_type':
            t = i.type.tensor_type.elem_type
            if t not in global_types and t not in special_types:
                cpp_type = tensorproto_type_to_cpp[t]
                print(f'Data type {cpp_type} of {description} is not supported.')
                is_supported = False
        else:
            # we don't support sequences, map, sparse tensors, or optional types in the pre-built package
            print(f'Data type {type_name} of {description} is not supported.')
            is_supported = False

        return is_supported

    def _input_output_is_supported(value_info, input_output):
        return _is_type_supported(value_info, f'graph {input_output} {value_info.name}')

    # node outputs are simpler to check.
    # node inputs have a much wider mix of types, some of which come from initializers and most likely are always
    # enabled as we generally do type reduction on the user data input to the operator and not the weights/etc. which
    # come from initializers.
    def _node_output_is_supported(name):
        is_supported = True
        if name in value_info_map:
            vi = value_info_map[name]
            is_supported = _is_type_supported(vi, f'node output {name}')
        else:
            # we don't have type info so ignore
            pass

        return is_supported

    for i in graph.input:
        if not _input_output_is_supported(i, 'input'):
            has_unsupported_types = True

    for i in graph.output:
        if not _input_output_is_supported(i, 'output'):
            has_unsupported_types = True

    for node in graph.node:
        # required_ops are map of [domain][opset] to set of op_type names. '' == ai.onnx
        domain = node.domain or 'ai.onnx'

        # special case Constant as we will convert to an initializer during model load
        if domain == 'ai.onnx' and node.op_type == 'Constant':
            continue

        # some models don't have complete imports. use 1 as a default as that's valid for custom domains and should
        # result in an error for any others. not sure why ONNX or ORT validation allows this though.
        opset = opsets[domain] if domain in opsets else 1
        if domain not in required_ops or \
                opset not in required_ops[domain] or \
                node.op_type not in required_ops[domain][opset]:
            unsupported_ops.add(f'{domain}:{opset}:{node.op_type}')

        for output_name in node.output:
            if not _node_output_is_supported(output_name):
                has_unsupported_types = True

        # recurse into subgraph for control flow nodes (Scan/Loop/If)
        for attr in node.attribute:
            if attr.HasField('g'):
                check_graph(attr.g, opsets, required_ops, global_types, special_types, unsupported_ops)

    return has_unsupported_types or unsupported_ops


def _get_global_tensorproto_types(op_type_impl_filter):
    '''
    Map the globally supported types (C++) to onnx.TensorProto.DataType values used in the model
    See https://github.com/onnx/onnx/blob/1faae95520649c93ae8d0b403816938a190f4fa7/onnx/onnx.proto#L485

    Additionally return a set of types we special case as being able to generally be considered as supported.
    :param op_type_impl_filter: type filter from reduced build configuration parser
    :return: tuple of globally enabled types and special cased types
    '''
    global_cpp_types = op_type_impl_filter.global_type_list()
    global_onnx_tensorproto_types = set()

    for t in global_cpp_types:
        if t in cpp_to_tensorproto_type:
            global_onnx_tensorproto_types.add(cpp_to_tensorproto_type[t])
        else:
            print(f'Error: Unexpected data type of {t}')
            sys.exit(-1)

    # a subset of operators require int32 and int64 to always be enabled, as those types are used for dimensions in
    # shapes and indices.
    # additionally we have a number of operators (e.g. Not, Where) that always require the use of bool.
    # this _may_ mean values involving these types can be processed, but without adding a lot more code we don't know
    # for sure.
    special_types = [cpp_to_tensorproto_type['int32_t'],
                     cpp_to_tensorproto_type['int64_t'],
                     cpp_to_tensorproto_type['bool']]

    return global_onnx_tensorproto_types, special_types


def main():
    parser = argparse.ArgumentParser(
        description='Check if model is likely to be able to be run using the ONNX Runtime Mobile Pre-Built Package',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    script_dir = os.path.dirname(os.path.realpath(__file__))

    # config that was used to create the pre-built package.
    # TODO: Would be nice to specify ORT release and pull the config for that release.
    default_config_path = \
        pathlib.Path(os.path.join(script_dir, '../ci_build/github/android/mobile_package.required_operators.config')
                     ).resolve()

    parser.add_argument('--config_path',
                        help='Path to required operators and types configuration used to build '
                             'the pre-built ORT mobile package.',
                        required=False,
                        type=pathlib.Path, default=default_config_path)

    parser.add_argument('model', help='Path to ONNX model to check', type=pathlib.Path)

    args = parser.parse_args()
    config_file = args.config_path.resolve(strict=True)  # must exist so strict=True
    model_file = args.model.resolve(strict=True)
    model = onnx.load(model_file)

    # we need to run shape inferencing to populate that type info for node outputs.
    # we will get warnings if the model uses ORT contrib ops, and type/shape inferencing will be lost downstream
    # of those. due to that we should also (or only) support ORT format models in this processing as they will have full
    # type/shape info for all ops.
    model_with_type_info = shape_inference.infer_shapes(model)

    # get type reduction and required ops from pre-built package config
    # we assume type reduction is enabled as that's what we use for our builds
    enable_type_reduction = True
    required_ops, op_type_impl_filter = reduced_build_config_parser.parse_config(config_file, enable_type_reduction)
    global_onnx_tensorproto_types, special_types = _get_global_tensorproto_types(op_type_impl_filter)

    # get the opset imports
    opsets = {}
    for entry in model.opset_import:
        # if empty it's ai.onnx
        domain = entry.domain or 'ai.onnx'
        opsets[domain] = entry.version

    unsupported_ops = set()
    print('Checking if the data types and operators used in the model are supported in the pre-built ORT package...\n')
    unsupported = check_graph(model_with_type_info.graph, opsets, required_ops,
                              global_onnx_tensorproto_types, special_types,
                              unsupported_ops)

    if unsupported_ops:
        print(' Unsupported operators:')
        for entry in sorted(unsupported_ops):
            print('  ' + entry)

    if unsupported:
        print('\nModel is not supported by the pre-built package due to unsupported types or operators.')
        print('Please see https://onnxruntime.ai/docs/reference/mobile/prebuilt-package/ for further details.')
    else:
        print('Model is most likely supported. '
              'Note that this check is not comprehensive so testing to validate is still required.')


if __name__ == '__main__':
    main()

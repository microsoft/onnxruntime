# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Helper script that will check if the types and operators used in an ONNX model
# would be supported by the pre-built ORT Mobile package.

import argparse
import onnx
import os
import pathlib
import sys
from util import reduced_build_config_parser


def check_graph(graph, opsets, required_ops, global_types, unsupported_ops):
    for i in graph.input:
        # check type in global types. warn if int32/int64 although those may be supported
        type_name = i.type.WhichOneof('value')
        if type_name == 'tensor_type':
            t = i.type.tensor_type.elem_type
            if t not in global_types:
                print(f'W: onnx.TensorProto.DataType input type of {t} may not be supported')
        else:
            # add warning
            print(f'W: Graph input type of {type_name} is not currently handled.')

    for node in graph.node:
        # required_ops are map of [domain][opset] to set of op_type names
        domain = node.domain
        if domain == '':
            domain = 'ai.onnx'

        # special case Constant as we will convert to an initializer during model load
        if domain == 'ai.onnx' and node.op_type == 'Constant':
            continue

        opset = opsets[domain]
        if domain not in required_ops or \
                opset not in required_ops[domain] or \
                node.op_type not in required_ops[domain][opset]:
            unsupported_ops.add(f'{domain}:{opset}:{node.op_type}')

        # recurse into subgraph for control flow nodes (Scan/Loop/If)
        for attr in node.attribute:
            if attr.HasField('g'):
                check_graph(attr.g, opsets, required_ops, global_types, unsupported_ops)

    return


# map the globally supported types (C++) to onnx.TensorProto.DataType values used in the model
# see https://github.com/onnx/onnx/blob/1faae95520649c93ae8d0b403816938a190f4fa7/onnx/onnx.proto#L485
def _get_global_tensorproto_types(op_type_impl_filter):
    global_cpp_types = op_type_impl_filter.global_type_list()
    global_onnx_tensorproto_types = set()
    cpp_to_tensorproto_types = {
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
        # COMPLEX64: 14 - not supported
        # COMPLEX128: 15 - not supported
        'BFloat16': 16
    }

    for t in global_cpp_types:
        if t in cpp_to_tensorproto_types:
            global_onnx_tensorproto_types.add(cpp_to_tensorproto_types[t])
        else:
            print(f'Error: Unexpected data type of {t}')
            sys.exit(-1)

    return global_onnx_tensorproto_types


def main():
    parser = argparse.ArgumentParser(
        description='Check if model can run using the ONNX Runtime Mobile Pre-Built Package',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    script_dir = os.path.dirname(os.path.realpath(__file__))

    # config that was used to create the pre-build package.
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

    # get type reduction and required ops from pre-built package config
    # we assume type reduction is enabled as that's what we use for our builds
    enable_type_reduction = True
    required_ops, op_type_impl_filter = reduced_build_config_parser.parse_config(config_file, enable_type_reduction)
    global_onnx_tensorproto_types = _get_global_tensorproto_types(op_type_impl_filter)

    # get the opset imports
    opsets = {}
    for entry in model.opset_import:
        domain = entry.domain
        if domain == '':
            domain = 'ai.onnx'
        opsets[domain] = entry.version

    unsupported_ops = set()
    check_graph(model.graph, opsets, required_ops, global_onnx_tensorproto_types, unsupported_ops)

    if unsupported_ops:
        print('Model is not supported by the pre-built package due to unsupported operators.')
        for entry in unsupported_ops:
            print(entry)

        print('\nPlease see https://onnxruntime.ai/docs/reference/mobile/prebuilt-package/ for further details.')


if __name__ == '__main__':
    main()

# SPDX-License-Identifier: Apache-2.0
# copied from https://github.com/onnx/onnx/blob/rel-1.11.0/onnx/backend/test/cmd_tools.py with modifications
import argparse
import json
import os
import shutil

import node as node_test
from onnx import numpy_helper
from typing import Text


TOP_DIR = os.path.realpath(os.path.dirname(__file__))
DATA_DIR = os.path.join(TOP_DIR, 'data')


def generate_data(args: argparse.Namespace) -> None:

    def prepare_dir(path: Text) -> None:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    cases = []
    cases += node_test.collect_testcases()    
    for case in cases:
        output_dir = os.path.join(
            args.output, case.kind, case.name)
        prepare_dir(output_dir)
        assert case.model
        with open(os.path.join(output_dir, 'model.onnx'), 'wb') as f:
            f.write(case.model.SerializeToString())
        assert case.data_sets
        for i, (inputs, outputs) in enumerate(case.data_sets):
            data_set_dir = os.path.join(
                output_dir, 'test_data_set_{}'.format(i))
            prepare_dir(data_set_dir)
            for j, input in enumerate(inputs):
                with open(os.path.join(
                        data_set_dir, 'input_{}.pb'.format(j)), 'wb') as f:
                    if case.model.graph.input[j].type.HasField('map_type'):
                        f.write(numpy_helper.from_dict(
                            input, case.model.graph.input[j].name).SerializeToString())
                    elif case.model.graph.input[j].type.HasField('sequence_type'):
                        f.write(numpy_helper.from_list(
                            input, case.model.graph.input[j].name).SerializeToString())
                    elif case.model.graph.input[j].type.HasField('optional_type'):
                        f.write(numpy_helper.from_optional(
                            input, case.model.graph.input[j].name).SerializeToString())
                    else:
                        assert case.model.graph.input[j].type.HasField('tensor_type')
                        f.write(numpy_helper.from_array(
                            input, case.model.graph.input[j].name).SerializeToString())
            for j, output in enumerate(outputs):
                with open(os.path.join(
                        data_set_dir, 'output_{}.pb'.format(j)), 'wb') as f:
                    if case.model.graph.output[j].type.HasField('map_type'):
                        f.write(numpy_helper.from_dict(
                            output, case.model.graph.output[j].name).SerializeToString())
                    elif case.model.graph.output[j].type.HasField('sequence_type'):
                        f.write(numpy_helper.from_list(
                            output, case.model.graph.output[j].name).SerializeToString())
                    elif case.model.graph.output[j].type.HasField('optional_type'):
                        f.write(numpy_helper.from_optional(
                            output, case.model.graph.output[j].name).SerializeToString())
                    else:
                        assert case.model.graph.output[j].type.HasField('tensor_type')
                        f.write(numpy_helper.from_array(
                            output, case.model.graph.output[j].name).SerializeToString())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('backend-test-tools')
    subparsers = parser.add_subparsers()

    subparser = subparsers.add_parser('generate-data', help='convert testcases to test data')
    subparser.add_argument('-o', '--output', default=DATA_DIR,
                           help='output directory (default: %(default)s)')
    subparser.add_argument('-t', '--op_type', default=None,
                           help='op_type for test case generation. (generates test data for the specified op_type only.)')
    subparser.set_defaults(func=generate_data)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
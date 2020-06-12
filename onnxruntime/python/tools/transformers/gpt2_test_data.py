#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------

# It is a tool to generate test data for a gpt2 model. The test data can be used in onnxruntime_perf_test.exe to evaluate the inference latency.

import sys
import argparse
import numpy as np
import os
from pathlib import Path
from onnx import numpy_helper
import onnxruntime


def create_test_data(model_path: str,
                     output_path: str,
                     test_cases: int,
                     dictionary_size: int = 10000,
                     num_heads: int = 12,
                     size_per_head: int = 64,
                     batch_size: int = 1,
                     sequence_length: int = 128,
                     num_layers: int = 12):

    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = onnxruntime.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])

    for test_case in range(test_cases):
        input_1 = np.random.randint(dictionary_size, size=(batch_size, 1), dtype=np.int64)
        tensor_1 = numpy_helper.from_array(input_1, 'input_ids')

        path = os.path.join(output_path, 'test_data_set_' + str(test_case))
        try:
            os.mkdir(path)
        except OSError:
            print("Creation of the directory %s failed" % path)
        else:
            print("Successfully created the directory %s " % path)

        input1_name = sess.get_inputs()[0].name
        output_names = [output.name for output in sess.get_outputs()]
        inputs = {input1_name: input_1}

        with open(os.path.join(path, 'input_{}.pb'.format(0)), 'wb') as f:
            f.write(tensor_1.SerializeToString())

        for i in range(num_layers):
            input_name = f"past_{i}"
            input = np.random.rand(2, batch_size, num_heads, sequence_length, size_per_head).astype(np.float32)
            tensor = numpy_helper.from_array(input, input_name)
            inputs.update({input_name: input})
            with open(os.path.join(path, 'input_{}.pb'.format(1 + i)), 'wb') as f:
                f.write(tensor.SerializeToString())

        result = sess.run(output_names, inputs)
        print("result 0 shape:", result[0].shape)
        print("result 1 shape:", result[1].shape)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', required=True, type=str, help="bert onnx model path.")

    parser.add_argument('--output_dir',
                        required=False,
                        type=str,
                        default=None,
                        help="output test data path. If not specified, .")

    parser.add_argument('--batch_size', required=False, type=int, default=1, help="batch size of input")

    parser.add_argument('--sequence_length',
                        required=False,
                        type=int,
                        default=128,
                        help="maximum sequence length of input")

    parser.add_argument('--samples', required=False, type=int, default=1, help="number of test cases to be generated")

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    output_dir = args.output_dir
    if output_dir is None:
        # Default output directory is a sub-directory under the directory of
        # model.
        p = Path(args.model)
        output_dir = os.path.join(p.parent, "{}_b{}_s{}".format(p.stem, args.batch_size, args.sequence_length))

    if output_dir is not None:
        # create the output directory if not existed
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
    else:
        print("Directory existed. test data files will be overwritten.")

    create_test_data(args.model,
                     output_dir,
                     test_cases=args.samples,
                     dictionary_size=10000,
                     num_heads=12,
                     size_per_head=64,
                     batch_size=args.batch_size,
                     sequence_length=args.sequence_length,
                     num_layers=12)

    print("Test data is saved to directory:", output_dir)


if __name__ == "__main__":
    main()

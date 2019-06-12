import argparse
import glob
import os
import sys

import numpy as np

from onnx import numpy_helper
from onnx import mapping
from onnx import TensorProto

def dump_pb_file(filename):
    "Read in a protobuf file or all .pb files in a directory, convert to numpy, and dump the data"

    with open(filename, 'rb') as f:
        tensor = TensorProto()
        tensor.ParseFromString(f.read())

    print("Name: {}".format(tensor.name))
    np_array = numpy_helper.to_array(tensor)
    print("Shape: {}".format(np_array.shape))
    print(np_array)

def dump_pb(dir_or_filename):
    "Read in a protobuf file or all .pb files in a directory, convert to numpy, and dump the data"

    if os.path.isdir(dir_or_filename):
        for f in glob.glob(os.path.join(dir_or_filename, '*.pb')):
            print(f)
            dump_pb_file(f)
    else:
        dump_pb_file(dir_or_filename)


def numpy_to_pb(name, np_data, out_filename):
    "Convert numpy data to protobuf"

    tensor = numpy_helper.from_array(np_data, name)
    with open(out_filename, 'wb') as f:
        f.write(tensor.SerializeToString())

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', help='Action to perform', choices=['dump_pb', 'numpy_to_pb'], required=True)
    parser.add_argument('--input', help='The input file', required=True)
    parser.add_argument('--name', help='The input/output name to use in the protobuf if action is numpy_to_pb')
    parser.add_argument('--output', help='The output file if action is numpy_to_pb')
    return parser


# import os
# def save_data(input_name, i, data, type='input'):
#     tensor = numpy_helper.from_array(data, input_name)
#     # TODO support multiple inputs
#
#     with open(os.path.join(r"D:\temp\good_loop\test_data_set_0", type + '_{}.pb'.format(i)), 'wb') as f:
#         f.write(tensor.SerializeToString())
#
# save_data("a", 0, a)
# save_data("b", 1, b)
# save_data("keep_going_inp", 2, keep_going)
# save_data("max_trip_count", 3, max_trip_count)
#
# save_data("b_loop", 0, np.array([6], dtype=np.float32), 'output')
# save_data("user_defined_vals", 1, np.array([-6, 12], dtype=np.float32).reshape(2, 1), 'output')


if __name__ == '__main__':

    print(sys.executable)
    print(numpy_helper)

    parser = get_arg_parser()
    args = parser.parse_args()

    if args.action == 'dump_pb':
        np.set_printoptions(precision=10)
        dump_pb(args.input)
    elif args.action == 'numpy_to_pb':
        # read data saved with
        data = np.load(args.input)
        numpy_to_pb(args.name, data, args.output)
    else:
        print("Unknown action.", file=sys.stderr)
        parser.print_help(sys.stderr)


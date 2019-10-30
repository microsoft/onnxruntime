import argparse
import glob
import os
import sys

import numpy as np

from onnx import numpy_helper
from onnx import mapping
from onnx import TensorProto


def dump_pb_file(filename):
    """Dump the data from a pb file containing a Tensor."""

    tensor = TensorProto()
    with open(filename, 'rb') as f:
        tensor.ParseFromString(f.read())

    print("Name: {}".format(tensor.name))

    np_array = numpy_helper.to_array(tensor)
    print("Shape: {}".format(np_array.shape))
    print(np_array)


def dump_pb(dir_or_filename):
    """Read in a protobuf file or all .pb files in a directory, convert to numpy, and dump the data."""

    if os.path.isdir(dir_or_filename):
        for f in glob.glob(os.path.join(dir_or_filename, '*.pb')):
            print(f)
            dump_pb_file(f)
    else:
        dump_pb_file(dir_or_filename)


def numpy_to_pb(name, np_data, out_filename):
    """Convert numpy data to a protobuf file."""

    tensor = numpy_helper.from_array(np_data, name)
    with open(out_filename, 'wb') as f:
        f.write(tensor.SerializeToString())


def image_to_numpy(filename, shape, channels_last, add_batch_dim):
    """Convert an image file into a numpy array."""

    import PIL.Image  # from 'Pillow' package

    img = PIL.Image.open(filename)
    if shape:
        img = img.resize(shape, PIL.Image.ANTIALIAS)
    img_as_np = np.array(img).astype(np.float32)
    if not channels_last:
        # HWC to CHW
        img_as_np = np.transpose(img_as_np, (2, 0, 1))

    if add_batch_dim:
        # to NCHW or NHWC
        img_as_np = np.expand_dims(img_as_np, axis=0)

    return img_as_np


def create_random_data(shape, type, minvalue, maxvalue, seed):
    nptype = np.dtype(type)
    np.random.seed(seed)
    return ((maxvalue - minvalue) * np.random.sample(shape) + minvalue).astype(nptype)


def update_name_in_pb(filename, name):
    """Update the name of the tensor in the pb file."""

    tensor = TensorProto()
    with open(filename, 'rb') as f_in:
        tensor.ParseFromString(f_in.read())
        tensor.name = name

    with open(filename, 'wb') as f_out:
        f_out.write(tensor.SerializeToString())


def get_arg_parser():
    parser = argparse.ArgumentParser(
        description=
        """
        Utilities for working with the input/output protobuf files used by onnx_test_runner.
        
        dump_pb: Dumps the tensor data for an individual protobuf pb file, or all pb files in a directory.
        numpy_to_pb: Write a numpy array to a protobuf file as a tensor.
        image_to_pb: Convert data from an image file into a protobuf file as a tensor.
        random_to_pb: Write random data into a protobuf file as a tensor.
        update_name_in_pb: Update the name of a tensor contained in a protobuf file.  
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--action', help='Action to perform',
                        choices=['dump_pb', 'numpy_to_pb', 'image_to_pb', 'random_to_pb', 'update_name_in_pb'],
                        required=True)
    parser.add_argument('--input', help='The input file or directory name')
    parser.add_argument('--name', help='The input/output name to use in the protobuf if action is '
                                       'numpy_to_pb, image_to_pb or update_name_in_pb')
    parser.add_argument('--output', help='The output file if action is numpy_to_pb, image_to_pb or random_to_pb')
    parser.add_argument('--shape', type=lambda s: [int(item) for item in s.split(',')],
                        help='If action==image_to_pb or random_to_pb provide the shape as comma separated values.'
                             ' e.g. --shape 200,200. Can be inferred from the image when using image_to_pb.')
    parser.add_argument('--channels_last', action='store_true',
                        help='If action==image_to_pb and the image needs to be transposed to channels last format.')
    parser.add_argument('--add_batch_dim', action='store_true',
                        help='If action==image_to_pb and the image needs the first dimension to be '
                             'the batch dim with value of 1.')
    parser.add_argument('--datatype', help="If action==random_to_pb provide the numpy dtype value for the data type. "
                                           "e.g. f4=float32, i8=int64. "
                                           "See: https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html")
    parser.add_argument('--min_value', default=0, type=int,
                        help="If action==random_to_pb limit the generated values to this minimum.")
    parser.add_argument('--max_value', default=1, type=int,
                        help="If action==random_to_pb limit the generated values to this maximum.")
    parser.add_argument('--seed', default=None, type=int,
                        help="If action==random_to_pb use this as the seed for the random values.")
    return parser


if __name__ == '__main__':
    arg_parser = get_arg_parser()
    args = arg_parser.parse_args()

    if args.action == 'dump_pb':
        if not args.input:
            print("Missing argument. Need input.", sys.stderr)
            sys.exit(-1)
        np.set_printoptions(precision=10)
        dump_pb(args.input)
    elif args.action == 'numpy_to_pb':
        if not args.input or not args.output or not args.name:
            print("Missing argument. Need input, output and name.", sys.stderr)
            sys.exit(-1)
        # read data saved with
        data = np.load(args.input)
        numpy_to_pb(args.name, data, args.output)
    elif args.action == 'image_to_pb':
        if not args.input or not args.output or not args.name:
            print("Missing argument. Need input, output, name.", file=sys.stderr)
            sys.exit(-1)

        img_np = image_to_numpy(args.input, args.shape, args.channels_last, args.add_batch_dim)
        numpy_to_pb(args.name, img_np, args.output)
    elif args.action == 'random_to_pb':
        if not args.output or not args.shape or not args.datatype or not args.name:
            print("Missing argument. Need output, shape, datatype and name.", file=sys.stderr)
            sys.exit(-1)

        data = create_random_data(args.shape, args.datatype, args.min_value, args.max_value, args.seed)
        numpy_to_pb(args.name, data, args.output)
    elif args.action == 'update_name_in_pb':
        if not args.input or not args.name:
            print("Missing argument. Need input and name.", file=sys.stderr)
            sys.exit(-1)

        update_name_in_pb(args.input, args.name)
    else:
        print("Unknown action.", file=sys.stderr)
        arg_parser.print_help(sys.stderr)


import argparse
import glob
import os
import sys

import numpy as np

import onnx
from onnx import numpy_helper
from onnx import mapping
from onnx import TensorProto

def dump_tensorproto_pb_file(filename):
    """Dump the data from a pb file containing a TensorProto."""

    tensor = onnx.load_tensor(filename)

    print("Name: {}".format(tensor.name))

    np_array = numpy_helper.to_array(tensor)
    print("Shape: {}".format(np_array.shape))
    print(np_array)


def dump_pb(dir_or_filename):
    """Dump the data from either a single .pb file, or all .pb files in a directory.
    All files must contain a serialized TensorProto."""

    if os.path.isdir(dir_or_filename):
        for f in glob.glob(os.path.join(dir_or_filename, '*.pb')):
            print(f)
            dump_tensorproto_pb_file(f)
    else:
        dump_tensorproto_pb_file(dir_or_filename)


def numpy_to_pb(name, np_data, out_filename):
    """Convert numpy data to a protobuf file."""

    tensor = numpy_helper.from_array(np_data, name)
    onnx.save_tensor(tensor, out_filename)


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


def update_name_in_pb(filename, name, output_filename):
    """Update the name of the tensor in the pb file."""

    tensor = onnx.load_tensor(filename)
    tensor.name = name

    if not output_filename:
        output_filename = filename

    onnx.save_tensor(tensor, output_filename)


def get_arg_parser():
    parser = argparse.ArgumentParser(
        description=
        """
        Utilities for working with the input/output protobuf files used by the ONNX test cases and onnx_test_runner. 
        These are expected to only contain a serialized TensorProto. 
        
        dump_pb: Dumps the TensorProto data from an individual pb file, or all pb files in a directory.
        numpy_to_pb: Convert numpy array saved to a file with numpy.save() to a TensorProto, and serialize to a pb file.
        image_to_pb: Convert data from an image file into a TensorProto, and serialize to a pb file.
        random_to_pb: Create a TensorProto with random data, and serialize to a pb file.
        update_name_in_pb: Update the TensorProto.name value in a pb file. 
                           Updates the input file unless --output <filename> is specified. 
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--action', help='Action to perform',
                        choices=['dump_pb', 'numpy_to_pb', 'image_to_pb', 'random_to_pb', 'update_name_in_pb'],
                        required=True)

    parser.add_argument('--input', help='The input filename or directory name')
    parser.add_argument('--name', help='The value to set TensorProto.name to if creating/updating one.')
    parser.add_argument('--output', help='Filename to serialize the TensorProto to.')

    image_to_pb_group = parser.add_argument_group('image_to_pb',
                                                  'image_to_pb specific options')
    image_to_pb_group.add_argument('--resize', default=None, type=lambda s: [int(item) for item in s.split(',')],
                                    help='Provide the shape as comma separated values to resize the image to.'
                                         ' e.g. --shape 200,200')
    image_to_pb_group.add_argument('--channels_last', action='store_true',
                                   help='Transpose image from channels first to channels last.')
    image_to_pb_group.add_argument('--add_batch_dim', action='store_true',
                                   help='Prepend a batch dimension with value of 1 to the shape. '
                                        'i.e. convert from CHW to NCHW')

    random_to_pb_group = parser.add_argument_group('random_to_pb',
                                                   'random_to_pb specific options')
    random_to_pb_group.add_argument('--shape', type=lambda s: [int(item) for item in s.split(',')],
                                    help='Provide the shape as comma separated values e.g. --shape 200,200')
    random_to_pb_group.add_argument('--datatype',
                                    help="numpy dtype value for the data type. e.g. f4=float32, i8=int64."
                                         "See: https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html")
    random_to_pb_group.add_argument('--min_value', default=0, type=int,
                                    help="Limit the generated values to this minimum.")
    random_to_pb_group.add_argument('--max_value', default=1, type=int,
                                    help="Limit the generated values to this maximum.")
    random_to_pb_group.add_argument('--seed', default=None, type=int,
                                    help="seed to use for the random values so they're deterministic.")

    return parser


if __name__ == '__main__':
    arg_parser = get_arg_parser()
    args = arg_parser.parse_args()

    if args.action == 'dump_pb':
        if not args.input:
            print("Missing argument. Need input to be specified.", file=sys.stderr)
            sys.exit(-1)
        np.set_printoptions(precision=10)
        dump_pb(args.input)
    elif args.action == 'numpy_to_pb':
        if not args.input or not args.output or not args.name:
            print("Missing argument. Need input, output and name to be specified.", file=sys.stderr)
            sys.exit(-1)
        # read data saved with
        data = np.load(args.input)
        numpy_to_pb(args.name, data, args.output)
    elif args.action == 'image_to_pb':
        if not args.input or not args.output or not args.name:
            print("Missing argument. Need input, output, name to be specified.", file=sys.stderr)
            sys.exit(-1)

        img_np = image_to_numpy(args.input, args.resize, args.channels_last, args.add_batch_dim)
        numpy_to_pb(args.name, img_np, args.output)
    elif args.action == 'random_to_pb':
        if not args.output or not args.shape or not args.datatype or not args.name:
            print("Missing argument. Need output, shape, datatype and name to be specified.", file=sys.stderr)
            sys.exit(-1)

        data = create_random_data(args.shape, args.datatype, args.min_value, args.max_value, args.seed)
        numpy_to_pb(args.name, data, args.output)
    elif args.action == 'update_name_in_pb':
        if not args.input or not args.name:
            print("Missing argument. Need input and name to be specified.", file=sys.stderr)
            sys.exit(-1)

        update_name_in_pb(args.input, args.name, args.output)
    else:
        print("Unknown action.", file=sys.stderr)
        arg_parser.print_help(sys.stderr)


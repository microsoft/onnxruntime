# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""
Creates an ONNX model with a single custom operator node that wraps an opaque model blob
meant to be executed by an external runtime. The model blob is serialized into the custom operator's
attributes.

Example:

python3 create_custom_op_wrapper.py --domain "test.domain"
        --custom_op_name my_custom_op
        --inputs "input0;FLOAT;1,10" "input1;FLOAT;1,10,10"
        --outputs "output0;STRING;10"
        --attribute_data xml googlenet-v1.xml
        --attribute_data bin googlenet-v1.bin
        -o test_model.onnx
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Union

import onnx
from onnx import TensorProto, helper

IO_NAME_INDEX = 0
IO_ELEM_TYPE_INDEX = 1
IO_SHAPE_INDEX = 2

TENSOR_TYPE_MAP = {
    "UNDEFINED": TensorProto.UNDEFINED,
    "FLOAT": TensorProto.FLOAT,
    "UINT8": TensorProto.UINT8,
    "INT8": TensorProto.INT8,
    "UINT16": TensorProto.UINT16,
    "INT16": TensorProto.INT16,
    "INT32": TensorProto.INT32,
    "INT64": TensorProto.INT64,
    "STRING": TensorProto.STRING,
    "BOOL": TensorProto.BOOL,
    "FLOAT16": TensorProto.FLOAT16,
    "DOUBLE": TensorProto.DOUBLE,
    "UINT32": TensorProto.UINT32,
    "UINT64": TensorProto.UINT64,
    "COMPLEX64": TensorProto.COMPLEX64,
    "COMPLEX128": TensorProto.COMPLEX128,
    "BFLOAT16": TensorProto.BFLOAT16,
    "FLOAT8E4M3FN": TensorProto.FLOAT8E4M3FN,
    "FLOAT8E4M3FNUZ": TensorProto.FLOAT8E4M3FNUZ,
    "FLOAT8E5M2": TensorProto.FLOAT8E5M2,
    "FLOAT8E5M2FNUZ": TensorProto.FLOAT8E5M2FNUZ,
}


@dataclass
class IOInfo:
    """
    Class that represents the index, name, element type, and shape of an input or output.
    """

    index: int
    name: str
    elem_type: TensorProto.DataType
    shape: Optional[List[Union[int, str]]]


def str_is_int(string: str) -> bool:
    try:
        int(string)
        return True
    except ValueError:
        return False


def parse_shape(shape_str: str) -> Optional[List[Union[int, str]]]:
    try:
        shape = [int(s) if str_is_int(s) else s for s in shape_str.split(",")]
    except ValueError:
        shape = None

    return shape


class ParseIOInfoAction(argparse.Action):
    def __call__(self, parser, namespace, io_strs, opt_str):
        is_input = opt_str == "--inputs"
        io_meta_name = "input" if is_input else "output"
        ios = []

        for io_idx, io_str in enumerate(io_strs):
            comp_strs = []

            try:
                comp_strs = io_str.split(";")
            except ValueError:  # noqa: PERF203
                parser.error(f"{opt_str}: {io_meta_name} info must be separated by ';'")

            if len(comp_strs) != 3:
                parser.error(f"{opt_str}: {io_meta_name} info must have 3 components, but provided {len(comp_strs)}.")

            # Get io name
            io_name = comp_strs[IO_NAME_INDEX]

            # Get io element type
            io_elem_type = TENSOR_TYPE_MAP.get(comp_strs[IO_ELEM_TYPE_INDEX])
            if io_elem_type is None:
                type_options = ",".join(TENSOR_TYPE_MAP.keys())
                parser.error(
                    f"{opt_str}: invalid {io_meta_name} element type '{comp_strs[IO_ELEM_TYPE_INDEX]}'. "
                    f"Must be one of {type_options}."
                )

            # Get io shape
            io_shape = parse_shape(comp_strs[IO_SHAPE_INDEX])
            if io_shape is None:
                parser.error(
                    f"{opt_str}: invalid {io_meta_name} shape '{comp_strs[IO_SHAPE_INDEX]}'. "
                    "Expected comma-separated list of integers."
                )

            ios.append(
                IOInfo(
                    index=io_idx,
                    name=io_name,
                    elem_type=io_elem_type,
                    shape=io_shape,
                )
            )

        # Sort ios on index
        ios = sorted(ios, key=lambda elem: elem.index)

        setattr(namespace, self.dest, ios)


def parse_arguments() -> argparse.Namespace:
    io_metavar = '"<name>;<elem_type>;<shape>"'
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-n",
        "--custom_op_name",
        required=True,
        help="The custom operator's name.",
    )

    parser.add_argument(
        "-d",
        "--domain",
        required=True,
        help="The ONNX domain name used by the custom operator node.",
    )

    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        action=ParseIOInfoAction,
        help="List of strings specifying the name, element type, and shape of every input (in order). "
        'Ex: --inputs "input_0;FLOAT;1,10" "input_1;INT8;32"',
        metavar=io_metavar,
    )

    parser.add_argument(
        "--outputs",
        nargs="+",
        required=True,
        action=ParseIOInfoAction,
        help="List of strings specifying the name, element type, and shape of every output (in order). "
        'Ex: --outputs "output_0;FLOAT;1,10" "output_1;INT8;32"',
        metavar=io_metavar,
    )

    parser.add_argument(
        "-o",
        "--output_model",
        required=True,
        help="The name of the generated output model.",
    )

    parser.add_argument(
        "-a",
        "--attribute_data",
        nargs=2,
        required=False,
        action="append",
        help="The attribute name and file path of the data to serialize as a node attribute. "
        "Can be provided multiple times. Ex: -a model_xml ./model.xml -a model_bin ./model.bin",
    )

    parser.add_argument(
        "-v",
        "--opset_version",
        type=int,
        required=False,
        default=13,
        help="The Opset version.",
    )

    return parser.parse_args()


def get_attributes(attr_data_info: List[List[str]]):
    if not attr_data_info:
        return {}

    attrs = {}

    for info in attr_data_info:
        filepath = os.path.normpath(info[1])

        if not os.path.exists(filepath):
            print(f"[ERROR] attribute file '{info[1]}' does not exist.", file=sys.stderr)
            sys.exit(1)

        data = b""
        with open(filepath, "rb") as file_desc:
            data = file_desc.read()

        tensor = helper.make_tensor(name=info[0], data_type=TensorProto.UINT8, dims=[len(data)], vals=data, raw=True)

        attrs[info[0]] = tensor

    return attrs


def main():
    args = parse_arguments()

    inputs = []
    input_names = []
    for inp in args.inputs:
        inputs.append(helper.make_tensor_value_info(inp.name, inp.elem_type, inp.shape))
        input_names.append(inp.name)

    outputs = []
    output_names = []
    for out in args.outputs:
        outputs.append(helper.make_tensor_value_info(out.name, out.elem_type, out.shape))
        output_names.append(out.name)

    attrs = get_attributes(args.attribute_data)

    custom_op_node = helper.make_node(
        args.custom_op_name,
        name=args.custom_op_name + "_0",
        inputs=input_names,
        outputs=output_names,
        domain=args.domain,
        **attrs,
    )

    output_model_path = os.path.normpath(args.output_model)
    graph_name, model_ext = os.path.splitext(os.path.basename(output_model_path))

    if model_ext != ".onnx":
        print(
            f"[ERROR] Invalid output model name '{output_model_path}'. Must end in '.onnx'",
            file=sys.stderr,
        )
        sys.exit(1)

    graph_def = helper.make_graph([custom_op_node], graph_name, inputs, outputs)
    model_def = helper.make_model(graph_def, opset_imports=[helper.make_opsetid(args.domain, args.opset_version)])

    onnx.checker.check_model(model_def)
    onnx.save(model_def, output_model_path)

    print(f"[INFO] Saved output model to {output_model_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
#
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import sys
import types
from importlib.machinery import SourceFileLoader

from opgen.generator import ORTGen
from opgen.parser import cpp_create_from_file as CPPParser
from opgen.writer import SourceWriter

parser = argparse.ArgumentParser(description="Generate ORT ATen operations")
parser.add_argument(
    "--ops_module", type=str, help="Python module containing the Onnx Operation signature and list of ops to map"
)
parser.add_argument("--output_file", default=None, type=str, help="Output file [default to std out]")
parser.add_argument("--header_file", type=str, help="Header file which contains ATen / Pytorch operation signature")
parser.add_argument(
    "--custom_ops", action="store_true", help="Whether we are generating code for custom ops or native operation"
)

args = parser.parse_args()
loader = SourceFileLoader("", args.ops_module)
ops_module = types.ModuleType(loader.name)
loader.exec_module(ops_module)

ortgen = ORTGen(
    ops_module.ops,
    type_promotion_ops=ops_module.type_promotion_ops,
    custom_ops=args.custom_ops,
    aten_output_type=ops_module.aten_output_type,
)

regdecs_path = args.header_file
print(f"INFO: Using RegistrationDeclarations from: {regdecs_path}")
output = sys.stdout
if args.output_file:
    output = open(args.output_file, "wt")

with CPPParser(regdecs_path) as parser, SourceWriter(output) as writer:
    ortgen.run(parser, writer)

#!/usr/bin/env python3
#
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from opgen.writer import SourceWriter as SourceWriter
from opgen.parser import cpp_create_from_file as CPPParser
import sys
import os
from opgen.generator import ORTGen as ORTGen
from importlib.machinery import SourceFileLoader
import argparse

parser = argparse.ArgumentParser(description='Generate ORT ATen operations')
parser.add_argument('--ops_module', type=str,
                    help='Python module containing the Onnx Operation signature and list of ops to map')
parser.add_argument('--output_file', default=None, type=str, help='Output file [default to std out]')
parser.add_argument('--header_file', type=str,
                    help='Header file which contains ATen / Pytorch operation signature')
parser.add_argument('--custom_ops', action='store_true', help='Whether we are generating code for custom ops or native operation')

args = parser.parse_args()
ops_module = SourceFileLoader("opgen.customop", args.ops_module).load_module()

ortgen = ORTGen(ops_module.ops, type_promotion_ops=ops_module.type_promotion_ops, custom_ops=args.custom_ops)

regdecs_path = args.header_file
print(f"INFO: Using ATen RegistrationDeclations from: {regdecs_path}")
output = sys.stdout
if args.output_file:
  output = open(args.output_file, 'wt')

with CPPParser(regdecs_path) as parser, SourceWriter(output) as writer:
  ortgen.run(parser, writer)

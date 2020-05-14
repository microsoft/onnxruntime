#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import glob
import logging
import os
import re

logging.basicConfig(format="[%(levelname)s] - %(message)s", level=logging.DEBUG)
log = logging.getLogger()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Find optimizers that involve operators which may need an update to the supported opset versions.')

    root_arg = parser.add_argument('--ort-root', '-o', required=True, type=str,
                                   help='The root directory of the ONNX Runtime repository to search.')

    args = parser.parse_args()

    if not os.path.isdir(args.ort_root):
        raise argparse.ArgumentError(root_arg, "{} is not a valid directory".format(args.ort_root))

    return args


def get_call_args_from_file(filename, function_or_declaration):
    """Search a file for all function calls or declarations that match the provided name.
    Currently requires both the opening '(' and closing ')' to be on the same line."""

    results = []
    with open(filename) as f:
        line_num = 0
        for line in f.readlines():
            for match in re.finditer(function_or_declaration, line):
                # check we have both the opening and closing brackets for the function call/declaration.
                # if we do we have all the arguments
                start = line.find('(', match.end())
                end = line.find(')', match.end())
                have_all_args = start != -1 and end != -1

                if have_all_args:
                    results.append(line[start + 1: end])
                else:
                    # TODO: handle automatically by merging lines
                    log.error("Call/Declaration is split over multiple lines. Please check manually."
                              "File:{} Line:{}".format(filename, line_num))
                    continue

            line_num += 1

    return results


def get_latest_op_versions(root_dir):
    """Find the entries for the latest opset for each operator."""

    op_to_opset = {}
    files = [os.path.join(root_dir, "onnxruntime/core/providers/cpu/cpu_execution_provider.cc"),
             os.path.join(root_dir, "onnxruntime/contrib_ops/cpu_contrib_kernels.cc")]

    for file in files:
        # e.g. class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, Clip);
        calls = get_call_args_from_file(file, 'ONNX_OPERATOR_KERNEL_CLASS_NAME')
        for call in calls:
            args = call.split(',')
            domain = args[1].strip()
            opset = args[2].strip()
            op = args[3].strip()
            op_to_opset[domain + '.' + op] = opset

        # e.g. class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, float, ArgMax);
        calls = get_call_args_from_file(file, 'ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME')
        for call in calls:
            args = call.split(',')
            domain = args[1].strip()
            opset = args[2].strip()
            op = args[4].strip()
            op_to_opset[domain + '.' + op] = opset

    return op_to_opset


def find_potential_issues(root_dir, op_to_opset):

    optimizer_dir = os.path.join(root_dir, "onnxruntime/core/optimizer")

    files = glob.glob(optimizer_dir + '/**/*.cc', recursive=True)
    files += glob.glob(optimizer_dir + '/**/*.h', recursive=True)

    for file in files:
        calls = get_call_args_from_file(file, 'graph_utils::IsSupportedOptypeVersionAndDomain')
        for call in calls:
            # Need to handle multiple comma separated version numbers, and the optional domain argument.
            # e.g. IsSupportedOptypeVersionAndDomain(node, "MaxPool", {1, 8, 10})
            #      IsSupportedOptypeVersionAndDomain(node, "FusedConv", {1}, kMSDomain)
            args = call.split(',', 2)  # first 2 args are simple, remainder need custom processing
            op = args[1].strip()
            versions_and_domain_arg = args[2]
            v1 = versions_and_domain_arg.find('{')
            v2 = versions_and_domain_arg.find('}')
            versions = versions_and_domain_arg[v1 + 1: v2].split(',')
            last_version = versions[-1].strip()

            domain_arg_start = versions_and_domain_arg.find(',', v2)
            if domain_arg_start != -1:
                domain = versions_and_domain_arg[domain_arg_start + 1:].strip()
            else:
                domain = "kOnnxDomain"

            if op.startswith('"') and op.endswith('"'):
                op = domain + '.' + op[1:-1]
            else:
                log.error("Symbolic name of '{}' found for op. Please check manually. File:{}".format(op, file))
                continue

            if op in op_to_opset:
                latest = op_to_opset[op]
                if int(latest) != int(last_version):
                    log.warning("Newer opset found for {}. Latest:{} Optimizer support ends at {}. File:{}"
                                .format(op, latest, last_version, file))
            else:
                log.error("Failed to find version information for {}. File:{}".format(op, file))


if __name__ == '__main__':
    arguments = parse_args()
    op_to_opset_map = get_latest_op_versions(arguments.ort_root)
    find_potential_issues(arguments.ort_root, op_to_opset_map)

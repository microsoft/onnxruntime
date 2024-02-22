#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import glob
import logging
import os
import re
import typing

logging.basicConfig(format="[%(levelname)s] - %(message)s", level=logging.DEBUG)
log = logging.getLogger()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find optimizers that involve operators which may need an update to the supported opset versions."
    )

    root_arg = parser.add_argument(
        "--ort-root", "-o", required=True, type=str, help="The root directory of the ONNX Runtime repository to search."
    )

    args = parser.parse_args()

    if not os.path.isdir(args.ort_root):
        raise argparse.ArgumentError(root_arg, f"{args.ort_root} is not a valid directory")

    return args


def get_call_args_from_file(filename: str, function_or_declaration: str) -> typing.List[str]:
    """
    Search a file for all function calls or declarations that match the provided name.
    Requires both the opening '(' and closing ')' to be on the same line.
    Handles multiple calls being on the same line.
    """

    results = []
    with open(filename) as f:
        line_num = 0
        for line in f:
            for match in re.finditer(function_or_declaration, line):
                # check we have both the opening and closing brackets for the function call/declaration.
                # if we do we have all the arguments
                start = line.find("(", match.end())
                end = line.find(")", match.end())
                have_all_args = start != -1 and end != -1

                if have_all_args:
                    results.append(line[start + 1 : end])
                else:
                    # TODO: handle automatically by merging lines
                    log.error(
                        "Call/Declaration is split over multiple lines. Please check manually."
                        f"File:{filename} Line:{line_num}"
                    )
                    continue

            line_num += 1

    return results


def get_multiline_call_args_from_file(filename: str, function_or_declaration: str) -> typing.List[str]:
    """
    Search a file for all function calls or declarations that match the provided name.
    Allows the opening '(' and closing ')' to be split across multiple lines.
    Supports a single call per line.
    """

    results = []
    with open(filename) as f:
        function_and_args = None

        for line in f:
            if not function_and_args:
                # look for new match
                start = line.find(function_or_declaration)
                if start != -1:
                    function_and_args = line[start:].strip()
            else:
                # append to existing line and look for closing ')'
                start = len(function_and_args)
                function_and_args += line.strip()

            if function_and_args:
                end = function_and_args.find(")", start)

                if end != -1:
                    start_args = function_and_args.find("(")
                    results.append(function_and_args[start_args + 1 : end])
                    function_and_args = None

    return results


def _add_if_newer(domain: str, op: str, opset: int, op_to_opset: typing.Dict[str, int]):
    key = domain + "." + op
    if key not in op_to_opset or op_to_opset[key] < opset:
        op_to_opset[key] = opset


def get_latest_ort_op_versions(root_dir):
    """Find the entries for the latest opset for each operator."""

    op_to_opset = {}
    files = [
        # for ONNX operators we use get_latest_onnx_op_versions
        # os.path.join(root_dir, "onnxruntime/core/providers/cpu/cpu_execution_provider.cc"),
        # for internal kernels we use the current registrations
        os.path.join(root_dir, "onnxruntime/contrib_ops/cpu/cpu_contrib_kernels.cc"),
        os.path.join(root_dir, "onnxruntime/contrib_ops/cuda/cuda_contrib_kernels.cc"),
    ]

    for file in files:
        # e.g. class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, Clip);
        calls = get_multiline_call_args_from_file(file, "ONNX_OPERATOR_KERNEL_CLASS_NAME")
        for call in calls:
            args = call.split(",")
            domain = args[1].strip()
            opset = args[2].strip()
            op = args[3].strip()
            _add_if_newer(domain, op, int(opset), op_to_opset)

        # e.g. class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 11, float, ArgMax);
        calls = get_multiline_call_args_from_file(file, "ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME")
        for call in calls:
            args = call.split(",")
            domain = args[1].strip()
            opset = args[2].strip()
            op = args[4].strip()
            _add_if_newer(domain, op, int(opset), op_to_opset)

    return op_to_opset


def get_latest_onnx_op_versions(root_dir):
    """Get the latest versions of the ONNX operators from the ONNX headers."""

    op_to_opset = {}
    files = [
        # operators with domain of 'Onnx'
        os.path.join(root_dir, "cmake/external/onnx/onnx/defs/operator_sets.h"),
        # ML operators with domain of 'OnnxML'
        os.path.join(root_dir, "cmake/external/onnx/onnx/defs/operator_sets_ml.h"),
    ]

    for file in files:
        # e.g. fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 17, LayerNormalization)>());
        #      fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxML, 3, TreeEnsembleClassifier)>());
        calls = get_multiline_call_args_from_file(file, "ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME")
        for call in calls:
            args = call.split(",")
            orig_domain = args[0].strip()
            # convert domain to the ORT constants
            domain = "kMLDomain" if orig_domain == "OnnxML" else "kOnnxDomain"
            opset = args[1].strip()
            op = args[2].strip()
            _add_if_newer(domain, op, int(opset), op_to_opset)

    return op_to_opset


def find_potential_issues(root_dir, op_to_opset):
    optimizer_dir = os.path.join(root_dir, "onnxruntime/core/optimizer")

    files = glob.glob(optimizer_dir + "/**/*.cc", recursive=True)
    files += glob.glob(optimizer_dir + "/**/*.h", recursive=True)

    for file in files:
        calls = get_call_args_from_file(file, "graph_utils::IsSupportedOptypeVersionAndDomain")
        for call in calls:
            # Need to handle multiple comma separated version numbers, and the optional domain argument.
            # e.g. IsSupportedOptypeVersionAndDomain(node, "MaxPool", {1, 8, 10})
            #      IsSupportedOptypeVersionAndDomain(node, "FusedConv", {1}, kMSDomain)
            args = call.split(",", 2)  # first 2 args are simple, remainder need custom processing
            op = args[1].strip()
            if not op.startswith('"') or not op.endswith('"'):
                log.error(f"Symbolic name of '{op}' found for op. Please check manually. File:{file}")
                continue

            versions_and_domain_arg = args[2]
            v1 = versions_and_domain_arg.find("{")
            v2 = versions_and_domain_arg.find("}")
            versions = versions_and_domain_arg[v1 + 1 : v2].split(",")
            last_version = versions[-1].strip()

            domain_arg_start = versions_and_domain_arg.find(",", v2)
            if domain_arg_start != -1:
                domain = versions_and_domain_arg[domain_arg_start + 1 :].strip()
            else:
                domain = "kOnnxDomain"

            op = domain + "." + op[1:-1]

            if op in op_to_opset:
                latest = op_to_opset[op]
                if int(latest) != int(last_version):
                    log.warning(
                        "Newer opset found for {}. Latest:{} Optimizer support ends at {}. File:{}".format(
                            op, latest, last_version, file
                        )
                    )
            else:
                log.error(f"Failed to find version information for {op}. File:{file}")


if __name__ == "__main__":
    arguments = parse_args()
    ort_to_opset_map = get_latest_ort_op_versions(arguments.ort_root)
    onnx_op_to_opset_map = get_latest_onnx_op_versions(arguments.ort_root)

    # merge the two maps
    op_to_opset_map = {**ort_to_opset_map, **onnx_op_to_opset_map}

    find_potential_issues(arguments.ort_root, op_to_opset_map)

# !/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
'''
Exclude operators that are unused/not required from build to reduce binary size.
'''

import argparse
import onnx
import op_registration_utils
import os
import shutil
import sys
import typing

from onnx import AttributeProto
from logger import get_logger


log = get_logger("exclude_unused_ops")


def _extract_ops_from_config(file_path, required_ops):
    '''extract ops from config file of format: domain;opset;op1,op2...'''

    if not file_path:
        return required_ops

    if not os.path.isfile(file_path):
        # exit. to continue may result in unexpectedly disabling all kernels.
        log.error('Configuration file {} does not exist'.format(file_path))
        sys.exit(-1)

    with open(file_path, 'r') as file_to_read:
        for stripped_line in [line.strip() for line in file_to_read.readlines()]:

            if not stripped_line:  # skip empty lines
                continue

            if stripped_line.startswith("#"):  # skip comments
                continue

            raw_domain, raw_opset, raw_ops = [segment.strip() for segment in stripped_line.split(';')]

            domain = op_registration_utils.map_domain(raw_domain)
            opset = int(raw_opset)
            operators = set([raw_op.strip() for raw_op in raw_ops.split(',')])

            if domain not in required_ops:
                required_ops[domain] = {opset: operators}
            elif opset not in required_ops[domain]:
                required_ops[domain][opset] = operators
            else:
                required_ops[domain][opset].update(operators)

    return required_ops  # end of _extract_ops_from_file(...)


def _extract_ops_from_graph(graph, operators, domain_opset_map):
    '''extract ops from graph and all subgraphs'''

    for operator in graph.node:
        mapped_domain = op_registration_utils.map_domain(operator.domain)

        if mapped_domain not in operators or mapped_domain not in domain_opset_map:
            continue

        operators[mapped_domain][domain_opset_map[mapped_domain]].add(operator.op_type)

        for attr in operator.attribute:
            if attr.type == AttributeProto.GRAPH:  # process subgraph
                _extract_ops_from_graph(attr.g, operators, domain_opset_map)


def _extract_ops_from_model(model_path, required_ops):
    '''extract ops from models under model_path and return a diction'''

    if not model_path:
        return required_ops

    if not os.path.isdir(model_path):
        # exit. to continue may result in unexpectedly disabling all kernels.
        log.error('Directory containing models does not exist: {}'.format(model_path))
        sys.exit(-1)

    for root, _, files in os.walk(model_path):
        for file in files:
            if file.endswith('.onnx'):
                model_path = os.path.join(root, file)
                model = onnx.load(model_path)
                domain_opset_map = {}

                if len(model.opset_import) == 0:
                    continue

                for opset in model.opset_import:
                    mapped_domain = op_registration_utils.map_domain(opset.domain)
                    domain_opset_map[mapped_domain] = opset.version

                    if mapped_domain not in required_ops:
                        required_ops[mapped_domain] = {opset.version: set()}

                    elif opset.version not in required_ops[mapped_domain]:
                        required_ops[mapped_domain][opset.version] = set()

                _extract_ops_from_graph(model.graph, required_ops, domain_opset_map)

    return required_ops  # end of _extract_ops_from_model(...)


class ExcludeOpsRegistrationProcessor(op_registration_utils.RegistrationProcessor):
    def __init__(self, required_ops, output_file):
        self.required_ops = required_ops
        self.output_file = output_file

    def _should_exclude_op(self, domain, operator, start_version, end_version):
        if domain not in self.required_ops:
            return True

        for opset in self.required_ops[domain]:
            if opset >= start_version and (end_version is None or opset <= end_version):
                if operator in self.required_ops[domain][opset]:
                    return False  # found a match, do not exclude

        return True

    def process_registration(self, lines: typing.List[str], domain: str, operator: str,
                             start_version: int, end_version: int = None, input_type: str = None):
        exclude = self._should_exclude_op(domain, operator, start_version, end_version)
        if exclude:
            log.info('Disabling {}:{}({})'.format(domain, operator, start_version))
            for line in lines:
                self.output_file.write('// ' + line)

            # edge case of last entry in table where we still need the terminating }; to not be commented out
            if lines[-1].rstrip().endswith('};'):
                self.output_file.write('};\n')
        else:
            for line in lines:
                self.output_file.write(line)

    def process_other_line(self, line):
        self.output_file.write(line)

    def ok(self):
        return True


def _exclude_unused_ops_in_registrations(required_operators, provider_registration_paths):
    '''rewrite provider registration file to exclude unused ops'''

    for kernel_registration_file in provider_registration_paths:
        if not os.path.isfile(kernel_registration_file):
            log.warning('Kernel registration file {} does not exist'.format(kernel_registration_file))
            return

        log.info("Processing {}".format(kernel_registration_file))

        backup_path = kernel_registration_file + '~'
        shutil.move(kernel_registration_file, backup_path)

        # read from backup and overwrite original with commented out lines for any kernels that are not required
        with open(kernel_registration_file, 'w') as file_to_write:
            processor = ExcludeOpsRegistrationProcessor(required_operators, file_to_write)

            op_registration_utils.process_kernel_registration_file(backup_path, processor)

            if not processor.ok():
                # error should have already been logged so just exit
                sys.exit(-1)


def _create_config_file_with_required_ops(required_operators, model_path, config_path, output_file):

    directory, filename = os.path.split(output_file)
    if not filename:
        log.error("Invalid path to write final config to. {}".format(output_file))
        sys.exit(-1)

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(output_file, 'w') as out:
        model_path_info = '--model_path {} '.format(model_path) if model_path else ''
        config_path_info = '--config_path {}'.format(config_path) if config_path else ''
        out.write("# Generated from {}{}\n".format(model_path_info, config_path_info))

        for domain in sorted(required_operators.keys()):
            if domain == 'UnknownDomain':
                continue

            # reverse the mapping of the domain. entry must exist given we created the required_operators dictionary.
            # also need to handle ai.onnx being special-cased as an empty string
            orig_domain = [key for (key, value) in op_registration_utils.domain_map.items() if value == domain][0]
            if not orig_domain:
                orig_domain = 'ai.onnx'

            for opset in sorted(required_operators[domain].keys()):
                ops = required_operators[domain][opset]
                if ops:
                    out.write("{};{};{}\n".format(orig_domain, opset, ','.join(sorted(ops))))

    log.info("Wrote set of required operators to {}".format(output_file))


def exclude_unused_ops(models_path, config_path, ort_root=None, use_cuda=True, output_config_path=None):
    '''Determine operators that are used, and either exclude them or create a configuration file that will.
    Note that this called directly from build.py'''

    if not models_path and not config_path:
        log.error('Please specify model_path and/or config_path.')
        sys.exit(-1)

    if not ort_root and not output_config_path:
        log.info('ort_root was not specified. Inferring ONNX Runtime repository root from location of this script.')

    required_ops = _extract_ops_from_config(config_path, _extract_ops_from_model(models_path, {}))

    if output_config_path:
        _create_config_file_with_required_ops(required_ops, models_path, config_path, output_config_path)
    else:
        registration_files = op_registration_utils.get_kernel_registration_files(ort_root, use_cuda)
        _exclude_unused_ops_in_registrations(required_ops, registration_files)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script to exclude unused operator kernels by disabling their registration in ONNXRuntime."
                    "See /docs/Reduced_Operator_Kernel_build.md for more information",
        usage="Provide model_path, config_path, or both, to disable kernel registration of unused kernels.")

    parser.add_argument(
        "--model_path", type=str, help="Path to folder containing one or more ONNX models")

    parser.add_argument(
        "--config_path", type=str, help="Path to configuration file with format of 'domain;opset;op1,op2...'")

    parser.add_argument(
        "--ort_root", type=str, help="Path to ONNXRuntime repository root. "
                                     "Inferred from the location of this script if not provided.")

    parser.add_argument(
        "--write_combined_config_to", type=str,
        help="Optional path to create a configuration file with the combined set of required kernels "
             "from processing --model_path and/or --config_path. If provided, a configuration file will be created "
             "and NO updates will be made to the kernel registrations."
    )

    args = parser.parse_args()

    models_path = os.path.abspath(args.model_path) if args.model_path else ''
    config_path = os.path.abspath(args.config_path) if args.config_path else ''
    ort_root = os.path.abspath(args.ort_root) if args.ort_root else ''

    if not models_path and not config_path:
        log.error('Please specify at least either model path or file path.')
        parser.print_help()
        sys.exit(-1)

    exclude_unused_ops(models_path, config_path, ort_root, use_cuda=True,
                       output_config_path=args.write_combined_config_to)

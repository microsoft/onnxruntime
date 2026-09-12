#!/usr/bin/env python
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import re

import onnxruntime.capi.onnxruntime_pybind11_state as rtpy


def main():
    cpu_info = {}
    for op in rtpy.get_all_opkernel_def():
        if op.provider != 'CPUExecutionProvider':
            continue
        if op.op_name not in cpu_info:
            cpu_info[op.op_name] = []
        if op.version_range not in cpu_info[op.op_name]:
            cpu_info[op.op_name].append(op.version_range)

    # print(cpu_info)

    webgpu_info = {}
    lines = open('../../onnxruntime/core/providers/js/js_execution_provider.cc').readlines()
    for line in lines:
        if not re.match('class ONNX_OPERATOR', line):
            continue
        items = re.search(r'\((.*)\)', line).group(1).replace(' ', '').split(',')

        if re.search('_t', items[-2]) or re.search('float', items[-2]):
            del items[-2]

        if items[-1] not in webgpu_info:
            webgpu_info[items[-1]] = []

        if len(items) == 4:
            range = (int(items[2]), 2147483647)
        elif len(items) in [5, 6]:
            range = (int(items[2]), int(items[3]))

        if range not in webgpu_info[items[-1]]:
            webgpu_info[items[-1]].append(range)

    # print(webgpu_info)

    for op in webgpu_info:
        if op in cpu_info and webgpu_info[op] != cpu_info[op]:
            print(f'=={op}==')
            print(f'CPU: {cpu_info[op]}')
            print(f'WebGPU: {webgpu_info[op]}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX Runtime Operator Diff Generator")
    args = parser.parse_args()
    main()

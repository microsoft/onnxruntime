'''comment out unused ops in cpu_execution_provider.cc by a set of models'''

import os
import onnx
from onnx import AttributeProto as AP

#pylint: disable=no-member,too-many-locals,too-many-statements

def extract_ops_from_model(model_path):
    '''extract ops from models under model_path and return a diction'''

    def map_domain(domain):
        if domain == 'ai.onnx.ml':
            return 'kMLDomain'
        return 'kOnnxDomain'

    def extract_ops_from_graph(graph, opsets, operators):
        '''extract ops from graph and all subgraphs'''

        for operator in graph.node:

            if operator.op_type not in operators:
                operators[operator.op_type] = {'opsets': {}, 'domains': {}}

            operators[operator.op_type]['domains'][map_domain(operator.domain)] = True

            for opset in opsets:
                operators[operator.op_type]['opsets'][opset.version] = True

            for attr in operator.attribute:

                if attr.type == AP.GRAPH: #process subgraph
                    extract_ops_from_graph(attr.g, opsets, operators)

                elif attr.type == AP.GRAPHS: #process all subgraphs
                    for subgraph in attr.graphs:
                        extract_ops_from_graph(subgraph, opsets, operators)

    referred_ops = {}
    for root, _, files in os.walk(model_path):
        for file in files:

            if file.endswith('.onnx'):
                model_path = os.path.join(root, file)
                model = onnx.load(model_path)
                extract_ops_from_graph(model.graph, model.opset_import, referred_ops)

    return referred_ops #end of extract_ops_from_model(...)


def rewrite_cpu_provider(model_path, file_path):
    '''rewrite provider file to exclude unused ops'''

    onnx_op = 'ONNX_OPERATOR_KERNEL_CLASS_NAME'
    onnx_op_len = len(onnx_op)
    onnx_typed_op = 'ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME'
    onnx_typed_op_len = len(onnx_typed_op)
    onnx_versioned_op = 'ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME'
    onnx_versioned_op_len = len(onnx_versioned_op)
    onnx_versioned_typed_op = 'ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME'
    onnx_versioned_typed_op_len = len(onnx_versioned_typed_op)
    version_map = {} #{op:{domain:[opset1, opset2, opset3 ...]}
    operators = extract_ops_from_model(model_path)

    def fill_version_map(op_type, opset_from, opset_to, domain):
        '''callback func to register op in version_map'''

        if op_type in version_map:
            if domain in version_map[op_type]:

                if opset_from not in version_map[op_type][domain]:
                    version_map[op_type][domain].append(opset_from)

                if opset_to not in version_map[op_type][domain]:
                    version_map[op_type][domain].append(opset_to)
            else:
                version_map[op_type][domain] =\
                    [opset_from, opset_to] if opset_from < opset_to else [opset_from]

        else:
            version_map[op_type] =\
                {domain: [opset_from, opset_to] if opset_from < opset_to else [opset_from]}

        return True #end of fill_version_map(...)


    def need_comment(op_type, opset_from, opset_to, domain):
        '''callback func to check if the op is in ops'''

        if op_type not in operators:
            return True
        if domain not in operators[op_type]['domains']:
            return True
        found_opset = False
        if opset_to == opset_from:
            offset = version_map[op_type][domain].index(opset_from)

            if offset >= len(version_map[op_type][domain]) - 1:
                opset_to = 100
            else:
                opset_to = version_map[op_type][domain][offset + 1]

        for opset in range(opset_from, opset_to+1):
            if opset in operators[op_type]['opsets']:
                found_opset = True
                break

        return not found_opset #end of need_comment(...)


    def process_lines(lines, offset, end_mark, call_back):
        '''extract op info from a logic code line start from offset to the line end
           with end_mark, then trigger callback(op_type, opset_from, opset_to, domain)
           return offset + num of lines processed
        '''

        lines_to_process = []
        while True: #collect the logical code line
            lines_to_process.append(lines[offset])
            stripped = lines[offset].strip()
            if stripped.endswith(end_mark):
                break
            offset += 1
        code_line = ''.join([line.strip() for line in lines_to_process])

        if onnx_op in code_line:
            trim_at = code_line.index(onnx_op) + onnx_op_len
            args = [arg.strip() for arg in code_line[trim_at: -len(end_mark)].split(',')]
            call_back_ret = call_back(args[-1], int(args[-2]), int(args[-2]), args[-3])

        elif onnx_typed_op in code_line:
            trim_at = code_line.index(onnx_typed_op) + onnx_typed_op_len
            args = [arg.strip() for arg in code_line[trim_at: -len(end_mark)].split(',')]
            call_back_ret = call_back(args[-1], int(args[-3]), int(args[-3]), args[-4])

        elif onnx_versioned_op in code_line:
            trim_at = code_line.index(onnx_versioned_op) + onnx_versioned_op_len
            args = [arg.strip() for arg in code_line[trim_at: -len(end_mark)].split(',')]
            call_back_ret = call_back(args[-1], int(args[-3]), int(args[-2]), args[-4])

        elif onnx_versioned_typed_op in code_line:
            trim_at = code_line.index(onnx_versioned_typed_op) + onnx_versioned_typed_op_len
            args = [arg.strip() for arg in code_line[trim_at: -len(end_mark)].split(',')]
            call_back_ret = call_back(args[-1], int(args[-4]), int(args[-3]), args[-5])

        return offset + 1, call_back_ret #end of process_lines(...)


    lines = []
    with open(file_path, 'r') as file_to_read:
        lines = file_to_read.readlines()

    os.rename(file_path, file_path + '.bak')
    with open(file_path, 'w') as file_to_write:
        line_index = 0

        while line_index < len(lines):

            line = lines[line_index]
            stripped = line.strip()

            if stripped.startswith('class ONNX_OPERATOR'):
                #collection versions of ops

                next_line_index, _ = process_lines(lines,
                                                   line_index,
                                                   ');',
                                                   fill_version_map)

                for index in range(line_index, next_line_index):
                    file_to_write.write(lines[index]) #leave as it was

                line_index = next_line_index

            elif stripped.startswith('BuildKernelCreateInfo'):
                #comment out unused ops

                next_line_index, disabled = process_lines(lines,
                                                          line_index,
                                                          ')>,',
                                                          need_comment)

                for index in range(line_index, next_line_index):
                    if disabled: #comment out unused
                        file_to_write.write('//' + lines[index])

                    else: #leave as it was
                        file_to_write.write(lines[index])

                line_index = next_line_index

            else: #leave as it was
                file_to_write.write(line)
                line_index += 1

    #end of rewrite_cpu_provider(...)
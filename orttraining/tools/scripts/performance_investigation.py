import argparse
import glob
import sys

import onnx

parser = argparse.ArgumentParser(description='ONNX file analyzer for performance investigation.')
parser.add_argument('onnx_file', type=str, nargs='?',
                    help='ONNX file to analyze; if empty, the program looks for ONNX files saved by ORTModule in the current dir')
args = parser.parse_args()

def process_file(onnx_file, torch_exported_file=False, pre_grad_optimized_file=False, optimized_file=False, execution_model_file=False):
    print(f"Processing {onnx_file}...")
    model = onnx.load(onnx_file)

    input_to_nodes = {}
    output_to_node = {}
    for node in model.graph.node:
        for i in node.input:
            input_to_nodes[i] = input_to_nodes.get(i, []) + [node]
        for o in node.output:
            output_to_node[o] = node

    aten_ops = []
    msgs = []

    visited = set([])
    def dfs(node, path):
        if node.name in visited:
            return
        visited.add(node.name)

        if node.op_type == "ATenOp":
            aten_ops.append(f"{node.name}: {node.attribute[0].s.decode('utf-8')}")

        # Search for 'memcpy' in the *_execution_model_<mode>.onnx.
        # In the ideal case, there should zero memcpy node in the final optimized training graph.
        if execution_model_file:
            if "Memcpy" in node.op_type:
                msgs.append(f"Memcpy node {node.name} found.")

        # Look for a node sandwiched by MemcpyToHost and MemcpyFromHost in the *_optimized_<mode>.onnx graph
        if optimized_file:
            if node.op_type == "MemcpyFromHost" and len(path) > 1 and path[-2].op_type == "MemcpyToHost":
                msgs.append(f"CUDA Kernel Missing for an Op {path[-1].op_type} for {path[-1].name} node.")

        # Look for node surrounded (sandwiched) by Casts node, see if ORT has already implemented fp16-saft kernels for them
        if optimized_file:
            if node.op_type == "Cast" and len(path) > 1 and path[-2].op_type == "Cast":
                msgs.append(f"Excessive casts around {path[-1].name} node.")        

        # Look for (Simplified)LayerNormalization in *_pre_grad_optimized_<mode>.onnx graph.
        # The layernorm subgraph (search for Pow node to begin with) should be fused into a single node.
        if pre_grad_optimized_file:
            if node.op_type == "LayerNormalization" or node.op_type == "SimplifiedLayerNormalization":
                for prev in path[::-1]:
                    if prev.op_type == "Pow":
                        msgs.append(f"Standalone {node.op_type} {node.name} found. The layernorm subgraph starting with {prev.name} should be fused into a single node.")
                    break

        # Look for (Fast)Gelu in *_pre_grad_optimized_<mode>.onnx graph.
        # The gelu subgraph (search for Erf node to begin with) should be fused into a single node.
        if pre_grad_optimized_file:
            if node.op_type == "Gelu" or node.op_type == "FastGelu":
                for prev in path[::-1]:
                    if prev.op_type == "Erf":
                        msgs.append(f"Standalone {node.op_type} {node.name} found. The gelu subgraph starting with {prev.name} should be fused into a single node.")
                    break

        # Look for stand-alone Dropout node in *_execution_model_<mode>.onnx graph.
        # Examine whether it should be fused with surrounding Add ops into BiasDropout node.
        if execution_model_file:
            if node.op_type == "Dropout" and path and path[-1].op_type == "Add":
                msgs.append(f"Examine whether {node.name} should be fused with the leading {path[-1].name} op into BiasDropout node.")

        # Look for stand-alone Softmax node in *_execution_model_<mode>.onnx graph.
        # Examine whether it should be fused with the leading Add ops into BiasSoftmax node.
        if execution_model_file:
            if node.op_type == "Softmax" and path and path[-1].op_type == "Add":
                msgs.append(f"Examine whether {node.name} should be fused with the leading {path[-1].name} op into BiasSoftmax node.")

        for o in node.output:
            for next_node in input_to_nodes.get(o, []):
                dfs(next_node, path + [node])

    for i in model.graph.input:
        if i.name in output_to_node:
            continue
        for node in input_to_nodes.get(i.name, []):
            if node.name not in visited:
                dfs(node, [])

    for i in model.graph.input:
        if i.name in output_to_node:
            continue
        for node in input_to_nodes.get(i.name, []):
            if node.name not in visited:
                dfs(node, [])

    if aten_ops:
        print("ATenOp(s) found:")
        print(10 * '-')
        for line in aten_ops:
            print(line)
        print(10 * '-')

    for line in msgs:
        print(line)


def main():
    if args.onnx_file:
        process_file(args.onnx_file, True, True, True, True)
    else:
        torch_exported_files = glob.glob("*_torch_exported_*.onnx")
        for file in torch_exported_files:
            process_file(file, torch_exported_file=True)

        pre_grad_optimized_files = glob.glob("*_pre_grad_optimized_*.onnx")
        for file in pre_grad_optimized_files:
            process_file(file, pre_grad_optimized_file=True)

        optimized_files = glob.glob("*_optimized_*.onnx")
        for file in optimized_files:
            if file not in pre_grad_optimized_files:
                process_file(file, optimized_file=True)

        execution_model_files = glob.glob("*_execution_model_file_*.onnx")
        for file in execution_model_files:
            process_file(file, execution_model_file=True)

if __name__ == "__main__":
    main()

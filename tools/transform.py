
import onnx
from onnx import helper
"""
TODO
- Skip MatMul to Transpose-Conv-Transpose if input is 4D
- Unsqueeze if input is <4D
- Squeeze if expected output is <4D
- Implement Reshape-Reducesum to Slice-Reducesum-Concat
- Dimensions are not matching in the result model
"""

def matmul_to_transpose_conv_transpose(model):
    # TODO skip 4D Matmul/Gemm
    cnt = 0
    graph = model.graph
    initializer_map =  {init.name: init for init in graph.initializer}
    nodes_to_remove = []
    for node in graph.node:
        if node.op_type == 'MatMul' or node.op_type == 'Gemm':
            for input in node.input:
                if input in initializer_map and len(initializer_map[input].dims) == 4:
                    continue
            nodes_to_remove.append(node)
    
    for node in nodes_to_remove:
        matmul_node_name = node.name
        graph.node.remove(node)
        transpose_before_node = helper.make_node(
            'Transpose',
            inputs=node.input,
            outputs=[matmul_node_name + '_transpose_before_output'],
            name=matmul_node_name + '_transpose_before'
        )
        conv_node = helper.make_node(
            'Conv',
            inputs=[matmul_node_name + '_transpose_before_output'],
            outputs=[matmul_node_name + '_transpose_output'],
            name=matmul_node_name + '_transpose'
        )
        transpose_after_node = helper.make_node(
            'Transpose',
            inputs=[matmul_node_name + '_transpose_output'],
            outputs=node.output,
            name=matmul_node_name + '_transpose_after'
        )
        graph.node.extend([transpose_before_node, conv_node, transpose_after_node])
        cnt += 1
    print(f"Replaced {cnt} MatMul nodes with Transpose-Conv-Transpose nodes")

# TODO      
# def reshape_reducesum_to_slice_reducesum_concat(model):

def qdq_to_clip(model):
    cnt = 0
    qualin_name_node_map, deqlin_name_node_map = {}, {}
    i = 0
    graph = model.graph
    for node in graph.node:
        if node.op_type =='QuantizeLinear':
            # check no output or if multiple connected nodes
            if not node.output[0] or len(node.output) > 1:
                continue
            qualin_name_node_map[node.output[0]] = node
        if node.op_type == 'DequantizeLinear':
            # print(node)
            if not node.input[0]:
                continue
            deqlin_name_node_map[node.input[0]] = node
    # print(qualin_name_node_map)
    
    nodes_to_remove = set()
    # clip_nodes = []
    input_replacement_map = {}
    for edge, qualin_node in qualin_name_node_map.items():
        if edge in deqlin_name_node_map:
            deqlin_node = deqlin_name_node_map[edge]
            # print(len(graph.node))
            graph.node.remove(qualin_node)
            graph.node.remove(deqlin_node)
            input_replacement_map[deqlin_node.output[0]] = qualin_node.input[0]
            # # print(len(graph.node))
            # clip_node = helper.make_node(
            #     'Clip',
            #     inputs=qualin_node.input,
            #     outputs=deqlin_node.output,
            #     name="Clip" + str(i),
            # )
            # graph.node.append(clip_node)
            i += 1
            cnt += 1
    # Find all nodes whose input is DequantizeLinear output
    for node in graph.node:
        for i, input_name in enumerate(node.input):
            if input_name in input_replacement_map:
                node.input[i] = input_replacement_map[input_name]
    print(f"Removed {cnt} QuantizeLinear and DequantizeLinear pairs")

def remove_deqlin(model):
    cnt = 0
    graph = model.graph
    initializer_names = set([init.name for init in graph.initializer])
    deqlin_output_initializer_mapping = {}
    nodes_to_remove = []
    for node in graph.node:
        if node.op_type == 'DequantizeLinear' and len(node.input) > 0 and node.input[0] in initializer_names:
            deqlin_output_initializer_mapping[node.output[0]] = node.input[0]
            nodes_to_remove.append(node)
        # Replace the node input after DequantizeLinear with initializer
        for i, input in enumerate(node.input):
            if input in deqlin_output_initializer_mapping:
                node.input[i] = deqlin_output_initializer_mapping[input]
    
    # Remove the nodes
    for node in nodes_to_remove:
        graph.node.remove(node)
        cnt += 1
    print(f"Removed {cnt} DequantizeLinear nodes")
            

model_path = './2024-12-13-PSH-v2.0.9/2024-12-13-PSH-v2.0.9/psapi_common_2_0_148197/PSH_ver_2.0.9.quant.onnx'
model = onnx.load(model_path)
qdq_to_clip(model)
remove_deqlin(model) # Needs to happen before matmul_to_transpose_conv_transpose
matmul_to_transpose_conv_transpose(model)
onnx.save(model, 'transformed_v2.onnx')

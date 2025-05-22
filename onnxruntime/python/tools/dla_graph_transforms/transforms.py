import onnx
from onnx import helper, TensorProto
import onnxscript
from onnxscript.rewriter import pattern
from onnxscript import ir
import numpy as np
from collections import defaultdict
from onnx import numpy_helper
import argparse
import subprocess
import re
import sys
import os
"""
ALL transform function starts with transform_ and called from transform_model.py.
ASSUMPTIONS:
- quantized values are either uint8 or uint16
Hardcoded:
- Transpose perm=[0,3,2,1]
- initializers to 4D
  - 1D [K] -> [1x1x1xK]
  - 2D [CxK] -> [KxCx1x1]
  - 3D insert 1 at 3rd dimension
- Reshape-ReduceSum to Slice-ReduceSum-Concat axes
- Expand 3D to 4D. Insert 1 at 0th dimension
- output Squeeze if Squeeze node exists axes = [0, 3]
- Flatten to reshape with axes [1, 1, 1, -1]
- Unsqueeze
    - From 3D axes=[1] 
    - From 2D axes=[0, -1]
- Squeeze
    - Update existing squeeze: [0, 3]
    - New squeeze: [1, 2]
- Gather
    - indices = [2]
- ReduceSum axes = [2] or keep [-1]
"""
###
# Private helper functions
###
def get_tensor_shape_map(graph_value_info):
    tensor_name_dim_map = {}
    for value_info in graph_value_info:
        tensor_type = value_info.type.tensor_type
        dims = 0
        if (tensor_type.HasField("shape")):
            for d in tensor_type.shape.dim:
                # the dimension may have a definite (integer) value or a symbolic identifier or neither:
                if d.HasField("dim_value"):
                    dims += 1
                else:
                    dims = -1
                    break
        tensor_name_dim_map[value_info.name] = dims
    return tensor_name_dim_map

def get_tensor_shape(graph_value_info, tensor_name):
    # print(f"Looking for tensor={tensor_name} in graph_value_info")
    # if tensor_name == "/model/transformer/decoder/layers.1/self_attn/Div_1_output_0":
    #     # print(graph_value_info)
    #     for value_info in graph_value_info:
    #         print(f"  - {value_info.name}")
    for value_info in graph_value_info:
        if value_info.name == tensor_name:
            print(f"Found tensor {tensor_name} in value_info value_info={value_info}")
            tensor_type = value_info.type.tensor_type
            if tensor_type.HasField("shape"):
                shape = []
                for dim in tensor_type.shape.dim:
                    if dim.HasField("dim_value"):
                        shape.append(dim.dim_value)
                    elif dim.HasField("dim_param"):
                        # For symbolic dimensions, you could use -1 or the parameter name
                        shape.append(-1)
                    else:
                        shape.append(-1)
                return shape
    return None  # Tensor name not found

def get_initializer_by_name(model, init_name):
    for init in model.graph.initializer:
        if init.name == init_name:
            return numpy_helper.to_array(init)
    return None

def calculate_clip_range(node, model):
    x_scale = get_initializer_by_name(model, node.input[1])
    x_zero_point = get_initializer_by_name(model, node.input[2])
    assert(x_scale, f"{node.name} should have x_scale value")
    int_max = np.int32(65535 if x_zero_point.dtype == np.uint16 else 255)
    if x_zero_point is None:
        print("x_zero_point is None!")
        x_zero_point = np.array(0, dtype=np.int32)
    else:
        x_zero_point = x_zero_point.astype(np.int32)
    clip_min = ((0 - x_zero_point) * x_scale).astype(np.float32)
    clip_max = ((int_max - x_zero_point) * x_scale).astype(np.float32)
    return clip_min, clip_max
###
# End of private helper functions
###
###
# Start of transform_* functions
###
def transform_matmul_to_transpose_conv_transpose(model):
    cnt = 0
    graph = model.graph
    tensor_name_dim_map = get_tensor_shape_map(graph.value_info)
    # print(f"tensor_name_dim_map={tensor_name_dim_map}")
    initializer_dim_map =  {init.name: len(init.dims) for init in graph.initializer}
    nodes_to_remove = []
    for node in graph.node:
        # if node.op_type == 'MatMul' or node.op_type == 'Gemm' and node.name not in ["/model/transformer/decoder/layers.1/self_attn/MatMul_3"]:
        if node.op_type == 'MatMul' or node.op_type == 'Gemm':
            need_transform = False
            all_inputs_3d = True
            for input in node.input:
                # Check if input is in initializer or value_info
                if input in initializer_dim_map:
                    dim = initializer_dim_map[input]
                elif input in tensor_name_dim_map:
                    dim = tensor_name_dim_map[input]
                else:
                    # If we can't determine the dimension, assume it needs transform
                    need_transform = True
                    all_inputs_3d = False
                    break
                    
                # Check if input is not 4D
                if dim != 4:
                    need_transform = True
                    
                # Check if input is not 3D
                if dim != 3:
                    all_inputs_3d = False
            
            # If all inputs are 3D, don't transform
            if all_inputs_3d:
                need_transform = False

            if need_transform:
                nodes_to_remove.append(node)
            else:
                print(f" Skipped MatMul/Gemm node {node.name} as both inputs are 4D")
    # print(f"nodes_to_remove={[node.name for node in nodes_to_remove]}")
    for node in nodes_to_remove:
        # print(f"input[0]={node.input[0]}")
        input0_shape = get_tensor_shape(graph.value_info, node.input[0])
        input1_shape = get_tensor_shape(graph.value_info, node.input[1])
        matmul_node_name = node.name
        graph.node.remove(node)
        skip_transpose = False
        # if node.name == "/model/transformer/decoder/layers.1/self_attn/MatMul_3":
        #     skip_transpose = True
        if skip_transpose:
            conv_inputs = [node.input[0], node.input[1]]
            if len(node.input) == 3: # Gemm has optional third input
                conv_inputs.append(node.input[2])
            conv_node = helper.make_node(
                'Conv',
                inputs=conv_inputs,
                outputs=node.output,
                name=matmul_node_name + '_conv'
            )
            if node.name == "/model/transformer/decoder/layers.1/self_attn/MatMul_3":
            # if input0_shape and input1_shape and input0_shape[0] != input1_shape[0]
                # if len(input0_shape) == 3 and input0_shape[2] > 1:  # If C_in > 1
                kernel_shape_attr = helper.make_attribute('kernel_shape', [1, 1])
                conv_node.attribute.append(kernel_shape_attr)

                # pads_attr = helper.make_attribute('pads', [0, 0, 0, 0])
                # conv_node.attribute.append(pads_attr)

                # strides_attr = helper.make_attribute('strides', [1, 1])
                # conv_node.attribute.append(strides_attr)
                print("Adding group attribute to Conv node")
                # Add group attribute to match channel count
                group_attr = helper.make_attribute('group', 8)
                conv_node.attribute.append(group_attr)
            graph.node.extend([conv_node])
        else:
            transpose_before_node = helper.make_node(
                'Transpose',
                inputs=[node.input[0]],

                outputs=[matmul_node_name + '_transpose_before_output'],
                name=matmul_node_name + '_transpose_before',
                perm=[0,3,2,1]
            )
            conv_inputs = [matmul_node_name + '_transpose_before_output', node.input[1]]
            if len(node.input) == 3: # Gemm has optional third input
                conv_inputs.append(node.input[2])
            conv_node = helper.make_node(
                'Conv',
                inputs=conv_inputs,
                outputs=[matmul_node_name + '_transpose_output'],
                name=matmul_node_name + '_conv'
            )
            print(f"input0_shape={input0_shape}")
            # if input0_shape[0] != input1_shape[0]:
            # group = input0_shape[0] / input1_shape[0]
            # kernel = [1,1]
            # if node.name == "/model/transformer/decoder/layers.1/self_attn/MatMul_3":
            # # if input0_shape and input1_shape and input0_shape[0] != input1_shape[0]
            #     # if len(input0_shape) == 3 and input0_shape[2] > 1:  # If C_in > 1
            #     kernel_shape_attr = helper.make_attribute('kernel_shape', [1, 1])
            #     conv_node.attribute.append(kernel_shape_attr)

            #     # pads_attr = helper.make_attribute('pads', [0, 0, 0, 0])
            #     # conv_node.attribute.append(pads_attr)

            #     # strides_attr = helper.make_attribute('strides', [1, 1])
            #     # conv_node.attribute.append(strides_attr)
            #     print("Adding group attribute to Conv node")
            #     # Add group attribute to match channel count
            #     group_attr = helper.make_attribute('group', 4)
            #     conv_node.attribute.append(group_attr)
            # if node.name == "/model/transformer/decoder/layers.1/self_attn/MatMul_4":
            # # if input0_shape and input1_shape and input0_shape[0] != input1_shape[0]
            #     # # if len(input0_shape) == 3 and input0_shape[2] > 1:  # If C_in > 1
            #     # kernel_shape_attr = helper.make_attribute('kernel_shape', [1, 1])
            #     # conv_node.attribute.append(kernel_shape_attr)

            #     # # pads_attr = helper.make_attribute('pads', [0, 0, 0, 0])
            #     # # conv_node.attribute.append(pads_attr)

            #     # # strides_attr = helper.make_attribute('strides', [1, 1])
            #     # # conv_node.attribute.append(strides_attr)
            #     # print("Adding group attribute to Conv node")
            #     # # Add group attribute to match channel count
            #     # group_attr = helper.make_attribute('group', 8)
            #     # conv_node.attribute.append(group_attr)
            #     conv_node.input[0], conv_node.input[1] = conv_node.input[1], conv_node.input[0]
            transpose_after_node = helper.make_node(
                'Transpose',
                inputs=[matmul_node_name + '_transpose_output'],
                outputs=node.output,
                name=matmul_node_name + '_transpose_after',
                perm=[0,3,2,1]
            )
            graph.node.extend([transpose_before_node, conv_node, transpose_after_node])
        cnt += 1
    print(f"Replaced {cnt} MatMul nodes with Transpose-Conv-Transpose nodes")

def transform_qdq_to_clip(model):
    cnt = 0
    qualin_name_node_map, deqlin_name_node_map = {}, {} # quantize_output_name -> quantize_node, dequantize_input_name -> dequantize_node
    i = 0
    graph = model.graph
    clip_range = {} # deq_input_name -> (clip_min, clip_max)
    for node in graph.node:
        if node.op_type =='QuantizeLinear':
            # check no output or if multiple connected nodes
            if not node.output[0] or len(node.output) > 1:
                continue
            qualin_name_node_map[node.output[0]] = node
        if node.op_type == 'DequantizeLinear':
            if not node.input[0]:
                continue
            assert len(node.input) == 3
            deqlin_name_node_map[node.input[0]] = node
            x_scale = get_initializer_by_name(model, node.input[1])
            x_zero_point = get_initializer_by_name(model, node.input[2])
            assert(x_scale, f"{node.name} should have x_scale value")
            int_max = np.int32(65535 if x_zero_point.dtype == np.uint16 else 255)
            if x_zero_point is None:
                print("x_zero_point is None!")
                x_zero_point = np.array(0, dtype=np.int32)
            else:
                x_zero_point = x_zero_point.astype(np.int32)
            clip_min = ((0 - x_zero_point) * x_scale).astype(np.float32)
            clip_max = ((int_max - x_zero_point) * x_scale).astype(np.float32)
            clip_range[node.input[0]] = (clip_min, clip_max)
    subgraph_output_input_map = {} # deqlin output -> qualin input
    clip_nodes_to_add = []
    for q_output, qualin_node in qualin_name_node_map.items():
        if q_output in deqlin_name_node_map:
            deqlin_node = deqlin_name_node_map[q_output]
            graph.node.remove(qualin_node)
            graph.node.remove(deqlin_node)
            subgraph_output_input_map[deqlin_node.output[0]] = qualin_node.input[0]
            clip_min, clip_max = clip_range[q_output]
            clip_min_init = numpy_helper.from_array(np.array([clip_min]), deqlin_node.name + '_clip_min')
            clip_max_init = numpy_helper.from_array(np.array([clip_max]), deqlin_node.name + '_clip_max')
            model.graph.initializer.extend([clip_min_init, clip_max_init])
            clip_node = helper.make_node(
                'Clip',
                inputs=[qualin_node.input[0], deqlin_node.name + '_clip_min', deqlin_node.name + '_clip_max'], # data, axes
                outputs=[deqlin_node.output[0]],
                name=deqlin_node.name + '_clip'
            )
            clip_nodes_to_add.append(clip_node)
            cnt += 1

    for clip_node in clip_nodes_to_add:
        graph.node.append(clip_node)
    print(f"Replaced {cnt} QuantizeLinear and DequantizeLinear pairs with Clip")
 
def transform_remove_qdq(model, keep_clip_after_inputs=False):
    q_output_to_q_node_map, dq_input_to_dq_node_map = {}, {} # q_output_name -> q_node, dq_input_name -> dq_node
    graph = model.graph
    
    # Collect all the candidate Q and DQ nodes
    for node in graph.node:
        if node.op_type =='QuantizeLinear':
            # check no output or if multiple connected nodes
            if not node.output[0] or len(node.output) > 1:
                continue
            q_output_to_q_node_map[node.output[0]] = node
        elif node.op_type == 'DequantizeLinear':
            if not node.input[0]:
                continue
            dq_input_to_dq_node_map[node.input[0]] = node

    qdq_node_pair_output_to_input_map = {} # qd output -> q input
    qdq_node_pair_input_to_output_map = {} # q input -> qd output
    clip_nodes_to_add = []
    cnt = 0

    # Find the Q and DQ node pairs and remove them.
    # There are following scenarios: 
    # 1) Node --> Q --> DQ --> Node'
    # 2) graph input --> Q --> DQ --> Node
    # 3) Node --> Q --> DQ --> graph output
    # 4) Node --> Q --> DQ ... --> Q --> DQ --> Node'
    for q_output, q_node in q_output_to_q_node_map.items():
        if q_output in dq_input_to_dq_node_map:
            dq_node = dq_input_to_dq_node_map[q_output]

            if keep_clip_after_inputs and q_node.input[0] in [graph_input.name for graph_input in model.graph.input]:
                # Calculate clip range if we want to keep the clip after inputs
                clip_min, clip_max = calculate_clip_range(dq_node, model)
                clip_min_init = numpy_helper.from_array(np.array([clip_min]), dq_node.name + '_clip_min')
                clip_max_init = numpy_helper.from_array(np.array([clip_max]), dq_node.name + '_clip_max')
                model.graph.initializer.extend([clip_min_init, clip_max_init])
                clip_node = helper.make_node(
                    'Clip',
                    inputs=[q_node.input[0], dq_node.name + '_clip_min', dq_node.name + '_clip_max'], # data, axes
                    outputs=[dq_node.output[0]],
                    name=dq_node.name + '_clip'
                )
                clip_nodes_to_add.append(clip_node)
            else:
                qdq_node_pair_output_to_input_map[dq_node.output[0]] = q_node.input[0]
                qdq_node_pair_input_to_output_map[q_node.input[0]] = dq_node.output[0]
            
            graph.node.remove(q_node)
            graph.node.remove(dq_node)
            cnt += 1      

    for clip_node in clip_nodes_to_add:
        graph.node.append(clip_node)

    # Make sure the predecessor and successor to the Q and DQ node pair are connected.
    # e.g. Node --> Q --> DQ --> Node'  =>  Node --> Node'
    for node in graph.node:
        for i, input_name in enumerate(node.input):
            # This node's ith input is from a DQ node.
            # 
            # Please note that the following while loop is for handling the #4 case mentioned above where
            # there could be multiple (ususally no more than 2) consecutive Q and DQ node pairs which are connected
            # e.g. Node --> Q --> DQ --> Q --> DQ --> Node'
            if input_name in qdq_node_pair_output_to_input_map:
                qdq_node_pair_output = input_name
                num_connected_qdq_node_pair = 0
                while True:
                    qdq_node_pair_input = qdq_node_pair_output_to_input_map[qdq_node_pair_output]
                    if qdq_node_pair_input not in qdq_node_pair_output_to_input_map:
                        node.input[i] = qdq_node_pair_input
                        break
                    else:
                        qdq_node_pair_output = qdq_node_pair_input
                    
                    num_connected_qdq_node_pair += 1
                    if num_connected_qdq_node_pair > 5:
                        print(f"Number of connected QDQ node pair is {num_connected_qdq_node_pair} which is not normal.")
                        sys.exit(1)

        for i, output_name in enumerate(node.output):
            # This node's ith output is consumed by a Q node.
            if output_name in qdq_node_pair_input_to_output_map:
                graph_output_candidate = qdq_node_pair_input_to_output_map[output_name]
                if graph_output_candidate in [output.name for output in graph.output]:
                    node.output[i] = graph_output_candidate

    print(f"Removed {cnt} QuantizeLinear and DequantizeLinear pairs")

def transform_remove_deqlin(model):
    def dequantize_initializer(deq_initializers, node_input, graph):
        for init in deq_initializers:
            if init.name == node_input[0]:
                x = numpy_helper.to_array(init).astype(np.int32)
            if init.name == node_input[1]:
                scale = numpy_helper.to_array(init).astype(np.float32)
            if init.name == node_input[2]:
                zero_point = numpy_helper.to_array(init).astype(np.int32)
        return ((x - zero_point) * scale).astype(np.float32) # Might have type issues: X, zero_point uint16, scale float32
    cnt = 0
    graph = model.graph
    initializer_names = set([init.name for init in graph.initializer])
    deqlin_output_initializer_mapping = {}
    nodes_to_remove = []
    for node in graph.node:
        if node.op_type == 'DequantizeLinear' and len(node.input) > 0 and node.input[0] in initializer_names:
            deq_initializers = []
            for init in graph.initializer:
                if init.name in node.input:
                    deq_initializers.append(init)
            dequantized_arr = dequantize_initializer(deq_initializers, node.input, graph).astype(np.float32)
            dequantized_init = numpy_helper.from_array(dequantized_arr, name=node.input[0])
            # remove all initializers in deq_initializers
            for init in deq_initializers:
                graph.initializer.remove(init)
            graph.initializer.append(dequantized_init)
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

"""
Add Unsqueeze node to model inputs if input is 2D or 3D. Special case is if there's already an Unsqueeze node
"""
def transform_non4d_model_inputs(model):
    cnt = 0
    graph = model.graph
    unsqueeze_to_add = []
    for graph_input in graph.input:
        dims = len(graph_input.type.tensor_type.shape.dim)
        if dims == 2 or dims == 3:
            unsqueeze_arr = np.array([0,-1], dtype=np.int64) if dims == 2 else np.array([1], dtype=np.int64)
            for node in graph.node:
                if len(node.input) > 0 and node.input[0] == graph_input.name:
                    if node.op_type == "Unsqueeze":
                        existing_unsqueeze_axes = None
                        for init in graph.initializer:
                            if init.name == node.input[1]:
                                existing_unsqueeze_axes = numpy_helper.to_array(init)
                                break
                        if existing_unsqueeze_axes is not None:
                            if len(existing_unsqueeze_axes) + dims == 4:
                                continue
                            new_unsqueeze_axes_name = node.name + "unsqueeze_axes_transformed"
                            unsqueeze_axes = numpy_helper.from_array(unsqueeze_arr, new_unsqueeze_axes_name)
                            node.input[1] = new_unsqueeze_axes_name
                            graph.initializer.append(unsqueeze_axes)
                    else:
                        new_unsqueeze_axes_name = graph_input.name + '_unsqueeze_axes'
                        unsqueezed_input_name = graph_input.name + '_unsqueeze_input'
                        unsqueeze_axes = numpy_helper.from_array(unsqueeze_arr, new_unsqueeze_axes_name)
                        graph.initializer.append(unsqueeze_axes)
                        unsqueeze_node = helper.make_node(
                            'Unsqueeze',
                            inputs=[graph_input.name, new_unsqueeze_axes_name],
                            outputs=[unsqueezed_input_name],
                            name=graph_input.name+'_unsqueeze',
                        )
                        unsqueeze_to_add.append(unsqueeze_node)
                        for node in graph.node:
                            for i, node_input_name in enumerate(node.input):
                                if node_input_name == graph_input.name:
                                    node.input[i] = unsqueezed_input_name
                    cnt += 1
    # Add all new unsqueeze nodes
    for node in unsqueeze_to_add:
        graph.node.append(node)
    print(f"Added/Updated {cnt} Unsqueeze node")

# Add Squeeze to non 4D model outputs
def transform_non4d_model_outputs(model):
    def is_squeeze_clip_output_pattern(squeeze_node, model):
        squeeze_output = squeeze_node.output[0]
        graph_output_names = [output.name for output in model.graph.output]
        for node in graph.node:
            if node.op_type == 'Clip' and node.input[0] == squeeze_output and node.output[0] in graph_output_names:
                return True
        return False
    cnt = 0
    graph = model.graph
    for graph_output in graph.output:
        update_existing_squeeze = False
        output_dim = len(graph_output.type.tensor_type.shape.dim)
        if output_dim < 4 and output_dim > 1:
            for node in graph.node:
                if node.op_type == 'Squeeze' and (node.output[0] == graph_output.name or is_squeeze_clip_output_pattern(node, model)):
                    update_existing_squeeze = True
                    squeeze_axes_name = graph_output.name + '_squeeze_axes'
                    if output_dim == 2:
                        squeeze_axes_arr = np.array([0, 3], dtype=np.int64)
                    squeeze_axes = numpy_helper.from_array(squeeze_axes_arr, squeeze_axes_name)
                    node.input[1] = squeeze_axes_name
                    graph.initializer.append(squeeze_axes)
            if not update_existing_squeeze:
                squeeze_input_name = graph_output.name + '_squeeze_input'
                squeeze_axes_name = graph_output.name + '_squeeze_axes'
                if output_dim == 2:
                    squeeze_axes_arr = np.array([0, 1], dtype=np.int64)
                elif output_dim == 3:
                    squeeze_axes_arr = np.array([2], dtype=np.int64)
                squeeze_axes = numpy_helper.from_array(squeeze_axes_arr, squeeze_axes_name)
                graph.initializer.append(squeeze_axes)
                squeeze_node = helper.make_node(
                    'Squeeze',
                    inputs=[squeeze_input_name, squeeze_axes_name],
                    outputs=[graph_output.name],
                    name=graph_output.name+'_squeeze',
                )
                # Change output of previous node to squeeze_input_name
                for node in graph.node:
                    for i, node_output_name in enumerate(node.output):
                        if node_output_name == graph_output.name:
                            node.output[i] = squeeze_input_name
                    # Handle intermediate nodes that have graph_output as input
                    for i, node_input_name in enumerate(node.input):
                        if node_input_name == graph_output.name:
                            node.input[i] = squeeze_input_name
                graph.node.append(squeeze_node)
                cnt += 1
    print(f"Added {cnt} Squeeze nodes for graph output")

# a) set keepdims=1 in the ReduceSum attributes, and b) increment the axes from [1] to [2]:
# TODO use -axes instead of fixed value
def transform_standalone_reducesum(model):
    graph = model.graph
    for node in graph.node:
        # A single ReduceSum, not transformed with reshape_reducesum_to_slice_reducesum_concat
        if node.op_type == 'ReduceSum' and not 'transformed' in node.name:
            # print(node.name)
            # print(node)
            # set keepdims = 1
            for attr in node.attribute:
                if attr.name == 'keepdims':
                    attr.i = 1
            axes_initializer = node.input[1] # axes
            initialize_to_remove = None
            new_reducesum_axes = None
            for initializer in graph.initializer:
                if initializer.name == axes_initializer:
                    current_axes = numpy_helper.to_array(initializer)
                    # Only set axes to [2] if current_axes is not [-1]
                    if not (current_axes.size == 1 and current_axes[0] == -1):
                        reducesum_axes_name = node.name + '_axes'
                        new_reducesum_axes = numpy_helper.from_array(np.array([2], dtype=np.int64), reducesum_axes_name)
                        initialize_to_remove = initializer
                        node.input[1] = reducesum_axes_name
            if new_reducesum_axes is not None:
                graph.initializer.append(new_reducesum_axes)

# Change Gather indices from scalar to vector, may need to update axis
def transform_gather(model):
    cnt = 0
    reshape_nodes_to_add = []
    reshape_inits_to_add = []

    # Create a mapping of node output names to their consumers
    output_to_consumers = {}
    for node in model.graph.node:
        for input_name in node.input:
            if input_name not in output_to_consumers:
                output_to_consumers[input_name] = []
            output_to_consumers[input_name].append(node)

    for node in model.graph.node:
        if node.op_type == 'Gather':
            cnt += 1
            indices_initializer_name = node.input[1]
            existing_indices = None
            for initializer in model.graph.initializer:
                if initializer.name == indices_initializer_name:
                    existing_indices = numpy_helper.to_array(initializer)
                    initializer_to_remove = initializer
                    break
            if existing_indices is not None:
                if existing_indices.ndim == 0:
                    indices_array = np.array([existing_indices.item()], dtype=existing_indices.dtype)
                else:
                    indices_array = existing_indices
            else:
                indices_array = np.array([0], dtype=np.int64)
            indices_initializer = numpy_helper.from_array(indices_array, name=indices_initializer_name)

            needs_reshape = False
            for attr in node.attribute:
                if attr.name == 'axis' and attr.i == 1:
                    attr.i = 2
                if attr.name == 'axis' and attr.i == 2:
                    attr.i = 3
                    needs_reshape = False
            # Remove the old initializer if it exists
            if initializer_to_remove is not None:
                model.graph.initializer.remove(initializer_to_remove)
            
            # Add the new initializer
            model.graph.initializer.append(indices_initializer)

            # Add reshape node if the axis was 2 (now 3)
            if needs_reshape:
                original_output = node.output[0]
                reshape_output = f"{original_output}_reshaped"
                
                # Create shape initializer for reshape
                reshape_shape_name = f"{node.name}_reshape_shape"
                reshape_shape = numpy_helper.from_array(
                    np.array([1, 1, 1, 3600], dtype=np.int64),
                    name=reshape_shape_name
                )
                reshape_inits_to_add.append(reshape_shape)
                
                # Create the reshape node
                reshape_node = helper.make_node(
                    'Reshape',
                    inputs=[original_output, reshape_shape_name],
                    outputs=[reshape_output],
                    name=f"{node.name}_reshape"
                )
                reshape_nodes_to_add.append(reshape_node)
                
                # Update all consumers of the original output
                for consumer in output_to_consumers.get(original_output, []):
                    for i, input_name in enumerate(consumer.input):
                        if input_name == original_output:
                            consumer.input[i] = reshape_output
                
                # Update graph outputs if necessary
                for output in model.graph.output:
                    if output.name == original_output:
                        output.name = reshape_output
    # Add all reshape nodes and initializers
    for node in reshape_nodes_to_add:
        model.graph.node.append(node)
        
    for initializer in reshape_inits_to_add:
        model.graph.initializer.append(initializer)
    print(f"Updated {cnt} Gather axis")

""" Change Gather indices from scalar to vector, may need to update axis
- 
"""
def transform_gatherelements(model):
    cnt = 0
    for node in model.graph.node:
        if node.op_type == 'GatherElements':
            cnt += 1
            indices_initializer_name = node.input[1]
            existing_indices = None
            for initializer in model.graph.initializer:
                if initializer.name == indices_initializer_name:
                    existing_indices = numpy_helper.to_array(initializer)
                    initializer_to_remove = initializer
                    break
            # If existing_indices is None, indices comes from another op, not the initializer, need to verify what happens in this case
            # if existing_indices is None:
            #     # print(f"GatherElements indices {initializer.name} shape: {existing_indices.ndim}")
            #     # if existing_indices.ndim == 2:
            #         # print(f"Reshaping 3D indices {initializer.name} from {existing_indices.shape} to 4D")
            #     indices_array = np.expand_dims([1, 1, 1, 200], axis=-1)
            # indices_initializer = numpy_helper.from_array(indices_array, name=indices_initializer_name)
            for attr in node.attribute:
                if attr.name == 'axis' and attr.i == 1:
                    attr.i = 3
            # Remove the old initializer if it exists
            # if initializer_to_remove is not None:
            #     model.graph.initializer.remove(initializer_to_remove)
            
            # Add the new initializer
            # model.graph.initializer.append(indices_initializer)
    print(f"Updated {cnt} GatherElements indices")

def transform_remove_intermediary_unsqueeze(model):
    """
    Remove all Unsqueeze operations that aren't directly connected to model inputs.
    This optimization removes unnecessary dimension expansion operations in the middle of the graph.
    """
    graph = model.graph
    
    # Get all input names
    input_names = set(input.name for input in graph.input)
    
    # Track nodes to remove
    nodes_to_remove = []
    
    # Create a mapping of node output names to their consumers
    output_to_consumers = {}
    for node in graph.node:
        for input_name in node.input:
            if input_name not in output_to_consumers:
                output_to_consumers[input_name] = []
            output_to_consumers[input_name].append(node)
    
    # Find all Unsqueeze nodes not directly connected to inputs
    for node in graph.node:
        if node.op_type == 'Unsqueeze' and node.input[0] not in input_names:
            # Get the input (source) of this Unsqueeze
            unsqueeze_input = node.input[0]
            
            # Get the output of this Unsqueeze
            unsqueeze_output = node.output[0]
            
            # Get all consumers of this Unsqueeze's output
            consumers = output_to_consumers.get(unsqueeze_output, [])
            
            # Rewire the graph to bypass this Unsqueeze
            for consumer in consumers:
                for i, input_name in enumerate(consumer.input):
                    if input_name == unsqueeze_output:
                        consumer.input[i] = unsqueeze_input
            
            # Mark this Unsqueeze for removal
            nodes_to_remove.append(node)
    
    # Remove the Unsqueeze nodes
    for node in nodes_to_remove:
        graph.node.remove(node)
    
    print(f"Removed {len(nodes_to_remove)} intermediary Unsqueeze operations")

def transform_remove_intermediary_squeeze(model):
    """
    Remove unnecessary Squeeze operations that are only used as intermediaries.
    This happens when a Squeeze operation is immediately followed by another operation
    and its output is not used anywhere else in the graph.
    
    Args:
        model: The ONNX model to transform
        
    Returns:
        The updated ONNX model
    """
    graph = model.graph
    
    # Create a mapping of tensor names to their producer nodes
    tensor_to_producer = {}
    for node in graph.node:
        for output in node.output:
            tensor_to_producer[output] = node
    
    # Create a mapping of tensor names to consumer nodes
    tensor_to_consumers = {}
    for node in graph.node:
        for input_name in node.input:
            if input_name not in tensor_to_consumers:
                tensor_to_consumers[input_name] = []
            tensor_to_consumers[input_name].append(node)
    
    # Find Squeeze nodes that can be removed
    squeeze_nodes_to_remove = []
    rewiring_map = {}  # Maps output tensor names to replacement input tensor names
    
    for node in graph.node:
        if node.op_type == 'Squeeze':
            # Check if this Squeeze is only used by one consumer
            if node.output[0] in tensor_to_consumers and len(tensor_to_consumers[node.output[0]]) == 1:
                # Make sure this Squeeze's output isn't a model output
                if not any(output.name == node.output[0] for output in graph.output):
                    squeeze_nodes_to_remove.append(node)
                    rewiring_map[node.output[0]] = node.input[0]
    
    # Update connections
    for node in graph.node:
        if node not in squeeze_nodes_to_remove:
            for i, input_name in enumerate(node.input):
                if input_name in rewiring_map:
                    node.input[i] = rewiring_map[input_name]
    
    # Remove the unnecessary Squeeze nodes
    for node in squeeze_nodes_to_remove:
        graph.node.remove(node)
    
    print(f"Removed {len(squeeze_nodes_to_remove)} intermediary Squeeze operations")

def transform_non4d_initializers(model):
    print(f"All initializer names = {[init.name for init in model.graph.initializer]}")
    # Purpose of need_to_expand_4D_init_names is to avoid expanding some 1D initializers such as axes, slice_begin
    need_to_expand_4D_init_names = []
    skip_init_names = []
    unary_dim_at_front_init_names = []
    for node in model.graph.node:
        if node.op_type in ['Div', 'Sub', 'Mul']:
            for input in node.input:
                need_to_expand_4D_init_names.append(input)
        if node.op_type in ['MatMul']:
            for input in node.input:
                skip_init_names.append(input)
        if node.op_type in ['Gemm']:
            for input in node.input:
                unary_dim_at_front_init_names.append(input)

    initializers_to_add = []
    initializer_to_remove = []
    for initializer in model.graph.initializer:
        if initializer.name.startswith("model.backbone.0.encoder.blocks.0.gamma_1_quantized"):
            print("In special branch for model.backbone.0.encoder.blocks.0.gamma_1_quantized")
            # 3D [192x1x1] -> [1x192x1x1]
            # initializer.dims.insert(0, 1)
        elif len(initializer.dims) == 1 and initializer.name in need_to_expand_4D_init_names:
            # 1D: [K] -> [1x1x1xK]
            initializer.dims.insert(0, 1)
            initializer.dims.insert(0, 1)
            initializer.dims.insert(0, 1)
        elif len(initializer.dims) == 2 and initializer.name in unary_dim_at_front_init_names:
            # 2D: [K, C] -> [1, 1, K, C]
            print(f"2D initializer unary_dim_at_front branch for {initializer.name}")
            new_dims = [1, 1] + list(initializer.dims)
            initializer.dims[:] = new_dims
            print(f"Dims after: {list(initializer.dims)}")
        elif len(initializer.dims) == 2 and initializer.name not in skip_init_names:
            # 2D [CxK] -> [KxCx1x1]
            c,k = initializer.dims[0], initializer.dims[1]
            init_arr = numpy_helper.to_array(initializer)
            transposed_arr = init_arr.T
            reshaped_arr = np.reshape(transposed_arr, (k, c, 1, 1))
            new_initializer = numpy_helper.from_array(reshaped_arr, initializer.name)
            initializer_to_remove.append(initializer)
            initializers_to_add.append(new_initializer)
            initializer.dims[0], initializer.dims[1] = k,c
            initializer.dims.insert(2, 1)
            initializer.dims.insert(3, 1)
            # new_dims = [1] + list(initializer.dims)
        elif len(initializer.dims) == 3:
            print(f"3D initializer found for {initializer.name}")
            print(f"Dims before: {list(initializer.dims)}")
            print(f"Inserting 1 at position 0")
            # initializer.dims.insert(0, 1)
            new_dims = [1] + list(initializer.dims)
            initializer.dims[:] = new_dims
            print(f"Dims after: {list(initializer.dims)}")
            data = numpy_helper.to_array(initializer)
            print(f"Actual shape: {data.shape}")

    [model.graph.initializer.remove(init) for init in initializer_to_remove]
    [model.graph.initializer.append(init) for init in initializers_to_add]
    
def transform_remove_all_tensor_value_shapes(model):
    for value_info in model.graph.value_info:
        tensor_type = value_info.type.tensor_type
        dims = 0
        if (tensor_type.HasField("shape")):
            tensor_type.ClearField("shape")

def transform_non4d_reshape(model):
    cnt = 0
    for node in model.graph.node:
        if node.op_type =='Reshape':
            # Check if axes
            for init in model.graph.initializer:
                if init.name == node.input[1]:
                    if init.dims[0] == 3:
                        # eg [-1, 512, 768] or [0,0,-1]
                        old_shape = numpy_helper.to_array(init)
                        init.dims[0] = 4
                        # insert at the first non-zero index
                        idx = next(i for i, v in enumerate(old_shape) if v != 0)
                        new_shape = np.insert(old_shape, idx, 1).astype(np.int64)
                        new_init = numpy_helper.from_array(new_shape, name=init.name)
                        model.graph.initializer.remove(init)
                        model.graph.initializer.append(new_init)
                        cnt += 1
    print(f"Updated {cnt} non4D Reshape nodes")

def transform_non4d_expand(model):
    cnt = 0
    for node in model.graph.node:
        if node.op_type == 'Expand':
            for init in model.graph.initializer:
                if init.name == node.input[1]:
                    if init.dims[0] == 3:
                        old_shape = numpy_helper.to_array(init)
                        init.dims[0] = 4
                        new_shape = np.insert(old_shape, 0, 1).astype(np.int64)
                        new_init = numpy_helper.from_array(new_shape, name=init.name)
                        model.graph.initializer.remove(init)
                        model.graph.initializer.append(new_init)
                        cnt += 1
    print(f"Updated {cnt} non4D Expand nodes")

"""
perm attribute of Transpose
- 2D: [T0, T1] -> [1, 1, T0 + 2, T1 + 2]
- 3D: [T0, T1, T2] -> [1, T0 + 1. T1 + 1, T2 + 1]
"""
def transform_non4d_transpose(model):
    cnt = 0
    for node in model.graph.node:
        if node.op_type == 'Transpose':
            for attr in node.attribute:
                if attr.name == 'perm' and len(attr.ints) == 2:
                    old_perm = list(attr.ints)
                    new_perm = [0, 1, old_perm[0] + 2, old_perm[1] + 2]
                    attr.ints[:] = new_perm
                    cnt += 1
                elif attr.name == 'perm' and len(attr.ints) == 3:
                    # [0, 2, 1] -> [0, 3, 2, 1]
                    old_perm = list(attr.ints)
                    new_perm = [0, old_perm[0] + 1, old_perm[1] + 1, old_perm[2] + 1]
                    attr.ints[:] = new_perm
                    cnt += 1
                
    print(f"Updated {cnt} non4D Transpose nodes")

# Transform Slice axes of non4D tensors
def transform_non4d_slice(model):
    cnt = 0
    for node in model.graph.node:
        # Skip transformed nodes
        if node.op_type == 'Slice' and not 'transformed_' in node.input[3]:
            new_init_to_add = None
            for init in model.graph.initializer:
                if init.name == node.input[3] and init.dims[0] == 1:
                    new_init_name = node.name + '_axes'
                    new_init_to_add = numpy_helper.from_array(np.array([-1], dtype=np.int64), name=new_init_name)
                    node.input[3] = new_init_name
                    cnt += 1
            if new_init_to_add is not None:
                model.graph.initializer.append(new_init_to_add) 
    print(f"Updated {cnt} non4D Slice axes")

# Transform LpNormalization axes of non4D tensors
def transform_non4d_lpnorm(model):
    cnt = 0
    for node in model.graph.node:
        if node.op_type == 'LpNormalization':
            for attr in node.attribute:
                if attr.name == 'axis':
                    attr.i = -1
            cnt += 1
    print(f"Updated {cnt} non4D LpNormalization axes")

# Flatten to Reshape
def transform_flatten(model):
    nodes_to_remove = []
    for node in model.graph.node:
        if node.op_type == 'Flatten':
            reshape_axes = numpy_helper.from_array(np.array([1, 1, 1, -1], dtype=np.int64), name=node.name + '_reshape_axes')
            reshape_node = helper.make_node(
                'Reshape',
                inputs=[node.input[0], reshape_axes.name],
                outputs=node.output,
                name=node.name + '_reshape'
            )
            nodes_to_remove.append(node)
            model.graph.initializer.append(reshape_axes)
            model.graph.node.append(reshape_node)
    # Remove flatten node(s)
    for node in nodes_to_remove:
        model.graph.node.remove(node)
    
# Debug function to add intermediate tensors to outputs
def transform_add_intermediate_tensors_to_outputs(model, intermediate_tensor_to_add=None):
    # Get existing output names
    existing_outputs = set(output.name for output in model.graph.output)
    
    # Collect all intermediate tensor names from node outputs
    if intermediate_tensor_to_add is None:
        for node in model.graph.node:
            for output in node.output:
                if output and output not in existing_outputs:
                    intermediate_tensor_to_add.add(output)  
    
    # Create ValueInfoProto for each intermediate tensor
    for tensor_name in intermediate_tensor_to_add:
        # Create a new output with default type FLOAT
        output_info = onnx.helper.make_tensor_value_info(
            tensor_name,
            onnx.TensorProto.FLOAT,  # Default to FLOAT type
            None  # Shape will be inferred if possible
        )
        model.graph.output.append(output_info)

def transform_remove_unused_initializers(model):
    """
    Remove initializers that are not used as inputs to any node in the graph.
    
    Args:
        model: An ONNX ModelProto object
        
    Returns:
        The modified model with unused initializers removed
    """
    graph = model.graph
    
    # Collect all node inputs
    used_inputs = set()
    for node in graph.node:
        used_inputs.update(node.input)
    
    # Also consider graph outputs as used
    for output in graph.output:
        used_inputs.add(output.name)
    
    # Find initializers that are used
    used_initializers = []
    removed_count = 0
    
    for initializer in graph.initializer:
        if initializer.name in used_inputs:
            used_initializers.append(initializer)
        else:
            removed_count += 1
    
    # Clear and reset initializers
    graph.ClearField("initializer")
    graph.initializer.extend(used_initializers)
    
    print(f"Removed {removed_count} unused initializers")

def transform_reducemax(model):
    """
    Transform ReduceMax operations:
    1. Update axes to [3]
    2. Set keepdims to 1
    3. Add a Reshape after ReduceMax with shape [1, 1, 1, 3600]
    """
    cnt = 0
    nodes_to_add = []
    reshape_shape_initializers = []
    
    # Create a mapping of node output names to their consumers
    output_to_consumers = {}
    for node in model.graph.node:
        for input_name in node.input:
            if input_name not in output_to_consumers:
                output_to_consumers[input_name] = []
            output_to_consumers[input_name].append(node)
    
    # Process each ReduceMax node
    for node in model.graph.node:
        if node.op_type == 'ReduceMax':
            cnt += 1
            
            # Update the axes attribute
            axes_attribute_found = False
            for attr in node.attribute:
                if attr.name == 'axes':
                    attr.ints[:] = [3]
                    axes_attribute_found = True
                elif attr.name == 'keepdims':
                    attr.i = 1
            
            # If no axes attribute found, add one
            if not axes_attribute_found:
                axes_attr = helper.make_attribute('axes', [3])
                node.attribute.append(axes_attr)
            
            # Create an initializer for the reshape shape
            reshape_shape_name = f"{node.name}_reshape_shape"
            reshape_shape = numpy_helper.from_array(
                np.array([1, 1, 1, 3600], dtype=np.int64),
                name=reshape_shape_name
            )
            reshape_shape_initializers.append(reshape_shape)
            
            # Create a reshape node
            reshape_output_name = f"{node.output[0]}_reshaped"
            original_output_name = node.output[0]
            
            reshape_node = helper.make_node(
                'Reshape',
                inputs=[original_output_name, reshape_shape_name],
                outputs=[reshape_output_name],
                name=f"{node.name}_reshape"
            )
            # nodes_to_add.append(reshape_node)
            
            # Update connections: all nodes that consumed the original ReduceMax output
            # # should now use the Reshape output
            # for consumer in output_to_consumers.get(original_output_name, []):
            #     for i, input_name in enumerate(consumer.input):
            #         if input_name == original_output_name:
            #             consumer.input[i] = reshape_output_name
            
            # # Update graph outputs if necessary
            # for output in model.graph.output:
            #     if output.name == original_output_name:
            #         output.name = reshape_output_name
    
    # Add all new nodes and initializers to the model
    for node in nodes_to_add:
        model.graph.node.append(node)
    
    for initializer in reshape_shape_initializers:
        model.graph.initializer.append(initializer)
    
    print(f"Updated {cnt} ReduceMax operations with Reshape")

def transform_slice_after_concat_axes(model):
    """
    Update axes of Slice nodes that follow Concat operations from [2] to [3].
    This transformation is part of adapting models from 3D to 4D tensor formats.
    
    Args:
        model: The ONNX model to transform
        
    Returns:
        The updated ONNX model
    """
    cnt = 0
    graph = model.graph
    
    # Track outputs of Concat nodes
    concat_outputs = set()
    for node in graph.node:
        if node.op_type == 'Concat':
            for output in node.output:
                concat_outputs.add(output)
    
    # Find Slice nodes that follow Concat and update their axes
    for node in graph.node:
        if node.op_type == 'Slice':
            # Check if this Slice takes input from a Concat and hasn't been transformed already
            if node.input[0] in concat_outputs and not 'transformed_' in node.input[3]:
                # Get the axes initializer (typically input[3])
                axes_initializer_name = node.input[3]
                initializer_to_remove = None
                
                # Find the axes initializer
                for initializer in graph.initializer:
                    if initializer.name == axes_initializer_name:
                        initializer_to_remove = initializer

                        # Check if current axes is [-1]
                        axes = numpy_helper.to_array(initializer)
                        if axes.size == 1 and axes[0] == -1:
                            # Create a new initializer with axes [2]
                            new_axes_name = node.name + '_transformed_axes'
                            new_initializer = numpy_helper.from_array(
                                np.array([3], dtype=axes.dtype),
                                name=new_axes_name
                            )
                            
                            # Update node to use new initializer
                            node.input[3] = new_axes_name
                            
                            # Add new initializer
                            graph.initializer.append(new_initializer)
                            if initializer_to_remove is not None:
                                model.graph.initializer.remove(initializer_to_remove)
                            cnt += 1
                            break
    
    print(f"Updated {cnt} Slice axes from [-1] to [2] after Concat operations")

def transform_topk_axis(model):
    """
    Update the 'axis' attribute of TopK operations.
    Transformations applied:
    - For 2D tensors with shape [1x3600]: change axis=1 to axis=2
    - For all other 2D tensors: change axis=1 to axis=-1
    
    Args:
        model: The ONNX model to transform
        
    Returns:
        The updated ONNX model
    """
    cnt_2d_3600 = 0
    cnt_2d_other = 0
    
    # Get tensor dimension information
    tensor_name_dim_map = get_tensor_shape_map(model.graph.value_info)
    
    for node in model.graph.node:
        if node.op_type == 'TopK':
            # Get the input tensor information
            input_name = node.input[0]
            input_shape = get_tensor_shape(model.graph.value_info, input_name)
            
            # Only process 2D tensors
            if input_shape and len(input_shape) == 2:
                # Special case for [1x3600] tensors, 2d->4D by adding leading and trailing unary dim,
                # change axis=1 to axis=2
                if input_shape[0] == 1 and input_shape[1] == 3600:
                    for attr in node.attribute:
                        if attr.name == 'axis' and attr.i == 1:
                            attr.i = 2
                            cnt_2d_3600 += 1
                            break
                # For all other 2D tensors
                else:
                    for attr in node.attribute:
                        if attr.name == 'axis' and attr.i == 1:
                            attr.i = -1
                            cnt_2d_other += 1
                            break
    
    print(f"Updated TopK operations:")
    print(f"  - {cnt_2d_3600} 2D [1x3600] inputs: axis 1 → 2")
    print(f"  - {cnt_2d_other} other 2D inputs: axis 1 → -1")

def transform_concat(model):
    """
    Transform Concat operations:
    1. Change axis from 2 to 3 (for 4D tensor operations)
    2. Add a Reshape operation after the Concat with shape [1, 1, 200, 4]
    3. If input tensor is 2D and axis==1, change axis to -1
    
    Args:
        model: The ONNX model to transform
        
    Returns:
        The updated ONNX model
    """
    cnt = 0
    cnt_2d = 0
    nodes_to_add = []
    reshape_initializers = []
    
    # Create a mapping of node output names to their consumers
    output_to_consumers = {}
    for node in model.graph.node:
        for input_name in node.input:
            if input_name not in output_to_consumers:
                output_to_consumers[input_name] = []
            output_to_consumers[input_name].append(node)
    
    # Get tensor dimension information
    tensor_name_dim_map = get_tensor_shape_map(model.graph.value_info)
    
    for node in model.graph.node:
        if node.op_type == 'Concat':
            # Check if this is a 2D input case
            input_is_2d = False
            if len(node.input) > 0:
                input_name = node.input[0]
                if input_name in tensor_name_dim_map and tensor_name_dim_map[input_name] == 2:
                    input_is_2d = True
            
            # Case: 2D tensor axis=1 -> 4D axis=-1
            if input_is_2d:
                for attr in node.attribute:
                    if attr.name == 'axis' and attr.i == 1:
                        attr.i = -1
                        cnt_2d += 1
                        break
            # Handle the specific Concat nodes case
            elif node.name in ["/model/transformer/Concat_12","/model/transformer/decoder/Concat_8"]:
                # Update the axis attribute
                axis_updated = False
                for attr in node.attribute:
                    if attr.name == 'axis' and attr.i == 2:
                        attr.i = 3
                        axis_updated = True
                        
                # If no axis attribute is found but Concat is present, add an axis attribute
                if not axis_updated and not any(attr.name == 'axis' for attr in node.attribute):
                    axis_attr = helper.make_attribute('axis', 3)
                    node.attribute.append(axis_attr)
                    axis_updated = False
                
                # If we updated the axis, add a reshape operation after the concat
                if axis_updated:
                    # Original concat output
                    original_output = node.output[0]
                    reshape_output = f"{original_output}_reshaped"
                    
                    # Create shape initializer for reshape
                    reshape_shape_name = f"{node.name}_reshape_shape"
                    reshape_shape = numpy_helper.from_array(
                        np.array([1, 1, 200, 4], dtype=np.int64),
                        name=reshape_shape_name
                    )
                    reshape_initializers.append(reshape_shape)
                    
                    # Create the reshape node
                    reshape_node = helper.make_node(
                        'Reshape',
                        inputs=[original_output, reshape_shape_name],
                        outputs=[reshape_output],
                        name=f"{node.name}_reshape"
                    )
                    nodes_to_add.append(reshape_node)
                    
                    # Update all consumers of the original concat output
                    for consumer in output_to_consumers.get(original_output, []):
                        for i, input_name in enumerate(consumer.input):
                            if input_name == original_output:
                                consumer.input[i] = reshape_output
                    
                    # Update graph outputs if necessary
                    for output in model.graph.output:
                        if output.name == original_output:
                            output.name = reshape_output
                    
                    cnt += 1
    
    # Add all new nodes and initializers to the model
    for node in nodes_to_add:
        model.graph.node.append(node)
    
    for initializer in reshape_initializers:
        model.graph.initializer.append(initializer)
    
    print(f"Updated {cnt} Concat operations from axis=2 to axis=3 with Reshape to [1,1,200,4]")
    print(f"Updated {cnt_2d} 2D Concat operations from axis=1 to axis=-1")

def transform_reshape_non4d_shape(model):
    """
    Transform Reshape operations with non-4D shapes by adding leading 1s.
    Also handle the special case of Reshape[-1,4] followed by Unsqueeze[1].
    This transfor needs to be runn before transform_intermediary_unsqueeze.
    
    Examples:
        [-1, 4] followed by Unsqueeze[1] -> [1, -1, 1, 4]
        
    Args:
        model: The ONNX model to transform
    """
    cnt = 0
    
    # Create a mapping of op outputs to their producers
    output_to_producer = {}
    for node in model.graph.node:
        for output in node.output:
            output_to_producer[output] = node
    
    # Create a mapping of op outputs to their consumers
    output_to_consumers = {}
    for node in model.graph.node:
        for input_name in node.input:
            if input_name not in output_to_consumers:
                output_to_consumers[input_name] = []
            output_to_consumers[input_name].append(node)
    
    # Track nodes to remove
    nodes_to_remove = []
    
    for node in model.graph.node:
        if node.op_type == 'Reshape':
            shape_initializer_name = node.input[1]
            shape_initializer = None
            
            # Find the shape initializer
            for initializer in model.graph.initializer:
                if initializer.name == shape_initializer_name:
                    shape_initializer = initializer
                    break
                    
            if shape_initializer:
                # Get the shape data
                shape_data = numpy_helper.to_array(shape_initializer)
                shape_dims = len(shape_data)
                
                # Special case: Check if this is a [-1, 4] Reshape followed by Unsqueeze[1]
                reshape_followed_by_unsqueeze = False
                unsqueeze_node = None
                
                if shape_dims == 2 and shape_data[0] == -1 and shape_data[1] == 4:
                    # Check if followed by an Unsqueeze
                    consumers = output_to_consumers.get(node.output[0], [])
                    for consumer in consumers:
                        if consumer.op_type == 'Unsqueeze' and len(consumer.input) > 1:
                            # Check the axes of the Unsqueeze
                            for initializer in model.graph.initializer:
                                if initializer.name == consumer.input[1]:
                                    axes = numpy_helper.to_array(initializer)
                                    if len(axes) == 1 and axes[0] == 1:
                                        reshape_followed_by_unsqueeze = True
                                        unsqueeze_node = consumer
                                        break
                            if reshape_followed_by_unsqueeze:
                                break
                
                if reshape_followed_by_unsqueeze:
                    # Create a new shape [1, -1, 1, 4] to replace both operations
                    new_shape = np.array([1, -1, 1, 4], dtype=shape_data.dtype)
                    new_initializer = numpy_helper.from_array(
                        new_shape, 
                        name=shape_initializer_name
                    )
                    
                    # Replace the old initializer
                    model.graph.initializer.remove(shape_initializer)
                    model.graph.initializer.append(new_initializer)
                    
                    # Rewire the graph to bypass the Unsqueeze
                    for consumer in output_to_consumers.get(unsqueeze_node.output[0], []):
                        for i, input_name in enumerate(consumer.input):
                            if input_name == unsqueeze_node.output[0]:
                                consumer.input[i] = node.output[0]
                    
                    # Mark the Unsqueeze for removal
                    nodes_to_remove.append(unsqueeze_node)
                    
                    cnt += 1       
    
    # Remove Unsqueeze nodes that were bypassed
    for node in nodes_to_remove:
        model.graph.node.remove(node)
    
    print(f"Transformed {cnt} Reshape[-1,4] + Unsqueeze[1] patterns to Reshape[1,-1,1,4]")

def transform_split_axis(model):
    """
    Transform Split operations:
    - When axis=1 and input tensor is 2D, change axis to -1
    - This is part of adapting models from 2D to 4D tensor formats
    
    Args:
        model: The ONNX model to transform
        
    Returns:
        The updated ONNX model
    """
    cnt = 0
    graph = model.graph
    
    # Create tensor dimension map
    tensor_name_dim_map = get_tensor_shape_map(graph.value_info)
 
    for node in graph.node:
        if node.op_type == 'Split':
            # Get input tensor dimensions
            input_name = node.input[0]
            input_dims = None
            
            if input_name in tensor_name_dim_map:
                input_dims = tensor_name_dim_map[input_name]
            
            # Only transform if input is 2D
            if input_dims == 2:
                # Update axis attribute
                for attr in node.attribute:
                    if attr.name == 'axis' and attr.i == 1:
                        attr.i = -1
                        cnt += 1
                        break
    
    print(f"Updated {cnt} Split operations from axis=1 to axis=-1 for 2D inputs")
    return model

###
# Onnxscript transform
###

"""
FROM
    x, shape
    |
Reshape axes
    |   /
ReduceSum
    |
reducesum_output
TO
      x
   /        \
Slice      Slice
  |           |
ReduceSum   ReduceSum
    \       /
    Concat
      |
reducesum_output
"""
def reshape_reducesum_pattern(op, x, shape, axes):
    reshape_output = op.Reshape(x, shape)
    reducesum_output = op.ReduceSum(reshape_output, axes)
    return reducesum_output

def slice_reducesum_concat(op, x, shape, axes):
    slice_0_starts = op.initializer(ir.tensor([0], dtype=ir.DataType.INT64, name=x.name + "_slice_0_starts"))
    slice_0_ends = op.initializer(ir.tensor([4], dtype=ir.DataType.INT64, name=x.name + "_slice_0_ends"))
    slice_1_starts = op.initializer(ir.tensor([4], dtype=ir.DataType.INT64, name=x.name + "_slice_1_starts"))
    slice_1_ends = op.initializer(ir.tensor([8], dtype=ir.DataType.INT64, name=x.name + "_slice_1_ends"))
    slice_reduce_axes = op.initializer(ir.tensor([3], dtype=ir.DataType.INT64, name=x.name + "_transformed_axes"))
    slice_output0 = op.Slice(x, slice_0_starts, slice_0_ends, slice_reduce_axes)
    slice_output1 = op.Slice(x, slice_1_starts, slice_1_ends, slice_reduce_axes)
    reducesum_output0 = op.ReduceSum(slice_output0, slice_reduce_axes)
    reducesum_output1 = op.ReduceSum(slice_output1, slice_reduce_axes)
    return op.Concat(reducesum_output0, reducesum_output1, axis=3)

def transform_reshape_reducesum(model):
    reshape_reducesum_rule = pattern.RewriteRule(reshape_reducesum_pattern, slice_reducesum_concat, verbose=10)
    model = onnxscript.rewriter.rewrite(
        model,
        pattern_rewrite_rules=[reshape_reducesum_rule],
    )
    return model
"""
FROM
    (x) 
    |
  Reshape
    |
   Clip
    |  
ReduceSum
    |
(reducesum_output)
TO
      (x)
   /        \
Slice      Slice
  |           |
Clip        Clip
  |           |
ReduceSum   ReduceSum
    \       /
    Concat
      |
(reducesum_output)
"""
def reshape_clip_reducesum_pattern(op, x, shape, clip_min, clip_max, axes):
    reshape_output = op.Reshape(x, shape)
    clip_output = op.Clip(reshape_output, clip_min, clip_max)
    return op.ReduceSum(clip_output, axes)

def slice_clip_reducesum_concat(op, x, shape, clip_min, clip_max, axes):
    slice_0_starts = op.initializer(ir.tensor([0], dtype=ir.DataType.INT64, name=x.name + "_slice_0_starts"))
    slice_0_ends = op.initializer(ir.tensor([4], dtype=ir.DataType.INT64, name=x.name + "_slice_0_ends"))
    slice_1_starts = op.initializer(ir.tensor([4], dtype=ir.DataType.INT64, name=x.name + "_slice_1_starts"))
    slice_1_ends = op.initializer(ir.tensor([8], dtype=ir.DataType.INT64, name=x.name + "_slice_1_ends"))
    slice_reduce_axes = op.initializer(ir.tensor([3], dtype=ir.DataType.INT64, name=x.name + "_transformed_axes"))
    slice_output0 = op.Slice(x, slice_0_starts, slice_0_ends, slice_reduce_axes)
    slice_output1 = op.Slice(x, slice_1_starts, slice_1_ends, slice_reduce_axes)
    clip_output0 = op.Clip(slice_output0, clip_min, clip_max)
    clip_output1 = op.Clip(slice_output1, clip_min, clip_max)
    reducesum_output0 = op.ReduceSum(clip_output0, slice_reduce_axes)
    reducesum_output1 = op.ReduceSum(clip_output1, slice_reduce_axes)
    return op.Concat(reducesum_output0, reducesum_output1, axis=3)

def transform_reshape_clip_reducesum(model):
    reshape_clip_reducesum_rule = pattern.RewriteRule(reshape_clip_reducesum_pattern, slice_clip_reducesum_concat, verbose=10)
    model = onnxscript.rewriter.rewrite(
        model,
        pattern_rewrite_rules=[reshape_clip_reducesum_rule],
    )
    return model

"""
FROM
data   
    |  
ReduceMax
---------
axes
keepdims
    |
reducemax_output
TO
data   
    |
ReduceMax
---------
axes=[3]
keepdims=1
    |    reshape_shape
    |   /
Reshape
    |
reducemax_output
"""
def reducemax_pattern(op, data):
    # return op.ReduceMax(data)
    return op.ReduceMax(data, _outputs=["orig_reducemax_output"])

def reducemax_reshape(op, data, orig_reducemax_output):
    reducemax_output = op.ReduceMax(data, axes=[3], keepdims=1)
    reshape_shape = op.initializer(ir.tensor([1,1,1,3600], dtype=ir.DataType.INT64, name=data.name + "_reshape_shape"))
    return op.Reshape(reducemax_output, reshape_shape)

def reducemax_not_transformed(
    context, orig_reducemax_output, **_
) -> bool:
    reducemax_node = orig_reducemax_output.producer()
    print(f"reducemax attributes: {reducemax_node.attributes}")
    print(f"reducemax_node.attributes['keepdims']= {reducemax_node.attributes['keepdims']}")
    print(f"reducemax_node.attributes['keepdims']==0 {reducemax_node.attributes['keepdims']==0}")
    print(f"reducemax_node.attributes['keepdims'].value==0 {reducemax_node.attributes['keepdims'].value==0}")
    return reducemax_node.attributes["keepdims"].value == 0

# def transform_reducemax(model):
#     reducemax_rule = pattern.RewriteRule(reducemax_pattern, reducemax_reshape, reducemax_not_transformed, verbose=10)
#     model = onnxscript.rewriter.rewrite(
#         model,
#         pattern_rewrite_rules=[reducemax_rule],
#     )
#     onnx.save(model, "PSA_debug.onnx")
#     return model

###
# public helper functions
###
def count_ops(model):
    ops = defaultdict(int)
    for node in model.graph.node:
        ops[node.op_type] += 1
        if node.op_type not in ops:
            ops.add(node.op_type)
    print("=== Graph ops count ===")
    for op_name, cnt in sorted(ops.items()):
        print(f"{cnt} {op_name}")

def execute_shape_inference(input_model, output_model):
    try:
        # Construct command for symbolic shape inference
        symbolic_shape_infer_cmd = (
            f"python ..\\onnxruntime\\onnxruntime\\python\\tools\\symbolic_shape_infer.py "
            f"--input {input_model} "
            f"--output {output_model} "
            # f"--auto_merge"
            f"--auto_merge --verbose 3"
        )
        
        # Run the command
        print(f"Running: {symbolic_shape_infer_cmd}")
        result = subprocess.run(symbolic_shape_infer_cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Shape inference completed successfully. Output saved to {output_model}")
            print(result.stdout)
        else:
            print("Shape inference failed with error:")
            print(result.stderr)
    except Exception as e:
        print(f"Error running symbolic shape inference: {str(e)}")

def all_tensors_are_4d(model):
    return False
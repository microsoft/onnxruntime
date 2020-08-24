# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###########################################################################

import itertools
import numpy as np
import onnx
from onnx import helper
from onnx import onnx_pb as onnx_proto


def _npfloat16_to_int(np_list):
    '''
    Convert numpy float16 to python int.
    :param np_list: numpy float16 list
    :return int_list: python int list
    '''
    return [int(bin(_.view('H'))[2:].zfill(16), 2) for _ in np_list]


def convert_tensor_float_to_float16(tensor):
    '''
    Convert tensor float to float16.
    :param tensor: TensorProto object
    :return tensor_float16: converted TensorProto object
    Example:
    ::
        from onnxmltools.utils.float16_converter import convert_tensor_float_to_float16
        new_tensor = convert_tensor_float_to_float16(tensor)
    '''
    if not isinstance(tensor, onnx_proto.TensorProto):
        raise ValueError('Expected input type is an ONNX TensorProto but got %s' % type(tensor))

    if tensor.data_type == onnx_proto.TensorProto.FLOAT:
        tensor.data_type = onnx_proto.TensorProto.FLOAT16
        # convert float_data (float type) to float16 and write to int32_data
        if tensor.float_data:
            int_list = _npfloat16_to_int(np.float16(tensor.float_data))
            tensor.int32_data[:] = int_list
            tensor.float_data[:] = []
        # convert raw_data (bytes type)
        if tensor.raw_data:
            # convert n.raw_data to float
            float32_list = np.fromstring(tensor.raw_data, dtype='float32')
            # convert float to float16
            float16_list = np.float16(float32_list)
            # convert float16 to bytes and write back to raw_data
            tensor.raw_data = float16_list.tostring()
    return tensor


def convert_float_to_float16(model):
    '''
    Convert tensor float type in the ONNX ModelProto input to tensor float16.
    :param model: ONNX ModelProto object
    :return: converted ONNX ModelProto object
    Examples:
    ::
        Example 1: Convert ONNX ModelProto object:
        from onnxmltools.utils.float16_converter import convert_float_to_float16
        new_onnx_model = convert_float_to_float16(onnx_model)
        Example 2: Convert ONNX model binary file:
        from onnxmltools.utils.float16_converter import convert_float_to_float16
        from onnxmltools.utils import load_model, save_model
        onnx_model = load_model('model.onnx')
        new_onnx_model = convert_float_to_float16(onnx_model)
        save_model(new_onnx_model, 'new_model.onnx')
    '''
    func_infer_shape = None
    if onnx.__version__ >= '1.2':
        try:
            from onnx.shape_inference import infer_shapes
            func_infer_shape = infer_shapes
        finally:
            pass

    if not isinstance(model, onnx_proto.ModelProto):
        raise ValueError('Expected model type is an ONNX ModelProto but got %s' % type(model))

    # create black list
    op_black_list = ['ArrayFeatureExtractor', 'Binarizer', 'CastMap', 'CategoryMapper', 'DictVectorizer',
                     'FeatureVectorizer', 'Imputer', 'LabelEncoder', 'LinearClassifier', 'LinearRegressor',
                     'Normalizer', 'OneHotEncoder', 'SVMClassifier', 'SVMRegressor', 'Scaler', 'TreeEnsembleClassifier',
                     'TreeEnsembleRegressor', 'ZipMap', 'NonMaxSuppression', 'TopK', 'RoiAlign', 'Resize',
                     'Range', 'CumSum', 'Upsample']
    input_of_op_black_list = []
    output_of_op_black_list = []
    op_need_to_modify = []
    # create a queue for BFS
    queue = []
    value_info_list = []
    node_list = []
    # type inference on input model
    if func_infer_shape is not None:
        model = func_infer_shape(model)
    queue.append(model)
    while queue:
        next_level = []
        for q in queue:
            # if q is model, push q.graph (GraphProto)
            if isinstance(q, onnx_proto.ModelProto):
                next_level.append(q.graph)
            # if q is model.graph, push q.node.attribute (AttributeProto)
            if isinstance(q, onnx_proto.GraphProto):
                for n in q.node:
                    # if n is in the black list (doesn't support float16), no conversion for the node,
                    # and save the node for further processing
                    if n.op_type in op_black_list:
                        input_of_op_black_list += n.input
                        output_of_op_black_list += n.output
                        node_list.append(n)
                    else:
                        if n.op_type == 'Cast':
                            for attr in n.attribute:
                                if attr.name == 'to' and attr.i == 1:
                                    attr.i = 10
                                    break
                        for attr in n.attribute:
                            next_level.append(attr)

                        # workaround.
                        # if input of op is output of another op in black list
                        # need to take care of this situation
                        if n.op_type == 'Concat':
                            for input in n.input:
                                if input in output_of_op_black_list:
                                    op_need_to_modify.append(n)
                                    break

            # if q is model.graph.node.attribute, push q.g and q.graphs (GraphProto)
            # and process node.attribute.t and node.attribute.tensors (TensorProto)
            if isinstance(q, onnx_proto.AttributeProto):
                next_level.append(q.g)
                for n in q.graphs:
                    next_level.append(n)
                q.t.CopyFrom(convert_tensor_float_to_float16(q.t))
                for n in q.tensors:
                    n = convert_tensor_float_to_float16(n)
            # if q is graph, process graph.initializer(TensorProto), input, output and value_info (ValueInfoProto)
            if isinstance(q, onnx_proto.GraphProto):
                for n in q.initializer:  # TensorProto type
                    if n.name in input_of_op_black_list:
                        continue
                    n = convert_tensor_float_to_float16(n)
                # for all ValueInfoProto with tensor(float) type in input, output and value_info, convert them to
                # tensor(float16) except map and seq(map). And save them in value_info_list for further processing
                for n in itertools.chain(q.input, q.output, q.value_info):
                    if n.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
                        n.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16
                        value_info_list.append(n)
        queue = next_level

    # process the nodes in black list that doesn't support tensor(float16)
    for node in node_list:
        # if input's name is in the value_info_list meaning input is tensor(float16) type,
        # insert a float16 to float Cast node before the node,
        # change current node's input name and create new value_info for the new name
        for i in range(len(node.input)):
            input = node.input[i]
            for value_info in value_info_list:
                if input == value_info.name:
                    # create new value_info for current node's new input name
                    new_value_info = model.graph.value_info.add()
                    new_value_info.CopyFrom(value_info)
                    output_name = node.name + '_input_cast_' + str(i)
                    new_value_info.name = output_name
                    new_value_info.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT
                    # add Cast node (from tensor(float16) to tensor(float) before current node
                    node_name = node.name + '_input_cast' + str(i)
                    new_node = [helper.make_node('Cast', [input], [output_name], to=1, name=node_name)]
                    model.graph.node.extend(new_node)
                    # change current node's input name
                    node.input[i] = output_name
                    continue
        # if output's name is in the value_info_list meaning output is tensor(float16) type, insert a float to
        # float16 Cast node after the node, change current node's output name and create new value_info for the new name
        for i in range(len(node.output)):
            output = node.output[i]
            for value_info in value_info_list:
                if output == value_info.name:
                    # create new value_info for current node's new output
                    new_value_info = model.graph.value_info.add()
                    new_value_info.CopyFrom(value_info)
                    input_name = node.name + '_output_cast_' + str(i)
                    new_value_info.name = input_name
                    new_value_info.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT
                    # add Cast node (from tensor(float) to tensor(float16) after current node
                    node_name = node.name + '_output_cast' + str(i)
                    new_node = [helper.make_node('Cast', [input_name], [output], to=10, name=node_name)]
                    model.graph.node.extend(new_node)
                    # change current node's input name
                    node.output[i] = input_name
                    continue

        for i in range(len(op_need_to_modify)):
            target_node = op_need_to_modify[i]

            for j in range(len(target_node.input)):
                target_input = target_node.input[j]

                if target_input not in output_of_op_black_list:
                    continue

                if target_input not in node.input and target_input not in node.output:
                    continue

                # create new value_info for current node's new input name
                from onnx import TensorProto
                x = helper.make_tensor_value_info('X', TensorProto.FLOAT, None)
                new_value_info = model.graph.value_info.add()
                new_value_info.CopyFrom(x)
                output_name = target_node.name + '_input_cast_' + str(i)
                new_value_info.name = output_name
                new_value_info.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16
                # add Cast node (from tensor(float) to tensor(float16) after current node
                node_name = target_node.name + '_output_cast' + str(i)
                new_node = [helper.make_node('Cast', [node.output[0]], [output_name], to=10, name=node_name)]
                model.graph.node.extend(new_node)
                # change current node's input name
                target_node.input[j] = output_name

                continue

    return model

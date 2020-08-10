# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# -*- coding: UTF-8 -*-
import argparse
from enum import Enum
import numpy as np
import onnx
from .node_factory import NodeFactory, ensure_opset
from .symbolic_shape_infer import SymbolicShapeInference, get_shape_from_type_proto

# trim outputs of LSTM/GRU/RNN if not used or outputed
def trim_unused_outputs(node, graph):
    trimmed = onnx.NodeProto()
    trimmed.CopyFrom(node)
    graph_outputs = [o.name for o in graph.output]
    for o_idx in range(len(node.output)):
        o = node.output[o_idx]
        use = [n for n in graph.node if o in list(n.input) + graph_outputs]
        if not use:
            trimmed.output[o_idx] = ''
    return trimmed

# squeeze init states, and split forward/reverse for bidirectional
def handle_init_state(init_state, nf, num_directions):
    if not init_state:
        return None
    if not nf.get_initializer(init_state) is None:
        return nf.get_initializer(init_state)
    if num_directions == 2:
        split_names = [init_state + '_split_0', init_state + '_split_1']
        nf.make_node('Split', init_state, {'axis':0}, split_names) # [1, batch, hidden]
        return [nf.make_node('Squeeze', s, {'axes':[0]}) for s in split_names]
    else:
        return [nf.make_node('Squeeze', init_state, {'axes':[0]})]

# handle some common attributes between LSTM/GRU/RNN
def handle_common_attributes(node, default_activations):
    direction = NodeFactory.get_attribute(node, 'direction')
    if direction:
        direction = str(direction, 'utf-8')
    else:
        direction = 'forward'
    num_directions = 2 if direction == 'bidirectional' else 1

    activations = NodeFactory.get_attribute(node, 'activations')
    if activations:
        activations = [str(x, 'utf-8').lower().capitalize() for x in activations]
    else:
        activations = default_activations * num_directions

    activation_alpha = NodeFactory.get_attribute(node, 'activation_alpha')
    activation_beta = NodeFactory.get_attribute(node, 'activation_beta')
    clip_threshold = NodeFactory.get_attribute(node, 'clip')
    # TODO: support these activation attributes
    assert not activation_alpha
    assert not activation_beta
    assert not clip_threshold
    return direction, num_directions, activations

# get batch_size, and create batch_node if needed
def handle_batch_size(X, nf, need_batch_node):
    X_vi = nf.get_value_info(X)
    assert X_vi
    dim = get_shape_from_type_proto(X_vi.type)[1]
    if type(dim) == str and need_batch_node:
        # only need to create batch_node for symbolic batch_size
        # otherwise, just use numpy.zeros
        X_shape = nf.make_node('Shape', X)
        node = nf.make_node('Slice', X_shape, {'axes':[0],'starts':[1],'ends':[2]})
    else:
        node = None
    return dim, node

# create default init state with zeros
def default_init_state(X, batch_size, batch_node, hidden_size, nf, postfix=''):
    if batch_node:
        shape = nf.make_node('Concat', [batch_node, np.asarray([hidden_size]).astype(np.int64)], {'axis':0})
        return nf.make_node('ConstantOfShape', shape)
    else:
        assert type(batch_size) == int
        # add default init state to graph input
        initializer_name = X + '_zero_init_state' + postfix
        initializer_shape = (batch_size, hidden_size)
        nf.make_value_info(initializer_name, onnx.TensorProto.FLOAT, initializer_shape, NodeFactory.ValueInfoType.input)
        return nf.make_initializer(np.zeros(initializer_shape, dtype=np.float32), initializer_name)

# declare seq_len_subgraph if needed
# note rank-1 for seq_len is to differentiate it from rank-2 states
def declare_seq_len_in_subgraph(seq_len, nf_body, prefix, batch_size):
    if seq_len:
        seq_len_subgraph = prefix + '_seq_len_subgraph'
        nf_body.make_value_info(seq_len_subgraph,
                                data_type=onnx.TensorProto.INT32,
                                shape=(batch_size,),
                                usage=NodeFactory.ValueInfoType.input)
    else:
        seq_len_subgraph = None
    return seq_len_subgraph

# hook subgraph outputs, with condition from seq_len_subgraph
def handle_subgraph_outputs(nf_body, seq_len_subgraph, batch_size, hidden_size, subgraph_output_or_default):
    final_subgraph_output = []
    if seq_len_subgraph:
        seq_len_output = nf_body.make_node('Sub', [seq_len_subgraph, np.asarray([1]).astype(np.int32)])
        nf_body.make_value_info(seq_len_output,
                                data_type=onnx.TensorProto.INT32,
                                shape=(batch_size,),
                                usage=NodeFactory.ValueInfoType.output)
        final_subgraph_output.append(seq_len_output)

        # since seq_len is rank-1, need to unsqueeze for Where op on rank-2 states
        condition = nf_body.make_node('Unsqueeze', nf_body.make_node('Greater', [seq_len_subgraph, np.zeros(shape=(), dtype=np.int32)]), {'axes':[1]})
        for valid, default in subgraph_output_or_default:
            final_subgraph_output.append(nf_body.make_node('Where', [condition, valid, default]))
    else:
        final_subgraph_output.append(None)
        for valid, default in subgraph_output_or_default:
            final_subgraph_output.append(nf_body.make_node('Identity', valid))

    for subgraph_o in final_subgraph_output[1:]:
        nf_body.make_value_info(subgraph_o,
                                data_type=onnx.TensorProto.FLOAT,
                                shape=(batch_size, hidden_size),
                                usage=NodeFactory.ValueInfoType.output)

    return final_subgraph_output

# unsqueeze/concat for the final outputs from scans, when the LSTM/GRU/RNN node is bidirectional
def handle_final_scan_outputs(node, nf, scan_outputs, state_outputs, num_directions):
    if num_directions == 2:
        def _bidirectional(outputs, axis, hook_output_name):
            outputs = [nf.make_node('Unsqueeze', x, {'axes':[axis]}) for x in outputs]
            nf.make_node('Concat', outputs, {'axis':axis}, output_names=hook_output_name)

        if node.output[0]:
            _bidirectional(scan_outputs, 1, node.output[0])
        for i_o in range(1, len(node.output)):
            _bidirectional(state_outputs[i_o - 1], 0, node.output[i_o])
    else:
        if node.output[0]:
            nf.make_node('Unsqueeze', scan_outputs[0], {'axes':[1]}, output_names=node.output[0])
        for i_o in range(1, len(node.output)):
            nf.make_node('Unsqueeze', state_outputs[i_o - 1], {'axes':[0]}, output_names=node.output[i_o])

def convert_lstm_to_scan(node, out_main_graph):
    assert node.op_type == 'LSTM'
    nf = NodeFactory(out_main_graph)
    with nf.scoped_prefix(node.output[0]) as scoped_prefix:
        X = node.input[0]
        Wa = nf.get_initializer(node.input[1])
        Ra = nf.get_initializer(node.input[2])
        num_inputs = len(node.input)
        Ba = nf.get_initializer(node.input[3]) if num_inputs > 3 else None
        seq_len = node.input[4] if num_inputs > 4 else None
        InitHa = node.input[5] if num_inputs > 5 else None
        InitCa = node.input[6] if num_inputs > 6 else None
        PB = node.input[7] if num_inputs > 7 else None

        # TODO: support peephole
        assert not PB

        direction, num_directions, activations = handle_common_attributes(node, ['Sigmoid', 'Tanh', 'Tanh'])

        hidden_size = NodeFactory.get_attribute(node, 'hidden_size')
        input_forget = NodeFactory.get_attribute(node, 'input_forget')

        # TODO: implement input_forget = 1
        assert not (input_forget != None and input_forget == 1)

        # split initializer if needed:
        is_same_init = InitHa == InitCa
        InitHa = handle_init_state(InitHa, nf, num_directions)
        if is_same_init:
            InitCa = InitHa
        else:
            InitCa = handle_init_state(InitCa, nf, num_directions)

        batch_size, batch_node = handle_batch_size(X, nf, InitHa is None or InitCa is None)

        scan_outputs = []
        scan_h_outputs = []
        scan_c_outputs = []
        for direction_index in range(num_directions):
            # for each direction
            # X [seq_len, batch_size, input_size]
            # W [4*hidden_size, input_size]
            # R [4*hidden_size, hidden_size]
            # B [8*hidden_size]
            # seq_len [batch_size]
            # init_h [batch_size, hidden_size]
            # init_c [batch_size, hidden_size]
            # PB [3*hidden_size]

            name_prefix = node.output[0] + '_' + str(direction_index) + '_'

            if InitHa is None:
                init_h = default_init_state(X, batch_size, batch_node, hidden_size, nf, '_H')
            else:
                init_h = InitHa[direction_index]

            if InitCa is None:
                init_c =  default_init_state(X, batch_size, batch_node, hidden_size, nf, '_C')
            else:
                init_c = InitCa[direction_index]

            input_size = Wa.shape[len(Wa.shape) - 1]
            Wt = np.transpose(Wa[direction_index])
            Rt = np.transpose(Ra[direction_index])
            B = Ba[direction_index].reshape(2, 4*hidden_size).sum(axis=0) # [4*hidden_size]
            X_proj = nf.make_node('MatMul', [X, Wt]) #[seq_len, batch_size, 4*hidden_size]
            X_proj = nf.make_node('Add', [X_proj, B])
            if num_directions == 1:
                is_backward = 0 if direction == 'forward' else 1
            else:
                is_backward = direction_index

            scan_body = onnx.GraphProto()
            scan_body.name = name_prefix + '_subgraph'

            nf_body = NodeFactory(out_main_graph, scan_body)
            with nf_body.scoped_prefix(name_prefix) as body_scoped_prefix:
                # subgraph inputs
                X_proj_subgraph = X_proj.name + '_subgraph'
                prev_h_subgraph = name_prefix + '_h_subgraph'
                prev_c_subgraph = name_prefix + '_c_subgraph'

                seq_len_subgraph = declare_seq_len_in_subgraph(seq_len, nf_body, X_proj.name, batch_size)

                for subgraph_i in [prev_h_subgraph, prev_c_subgraph]:
                    nf_body.make_value_info(subgraph_i,
                                            data_type=onnx.TensorProto.FLOAT,
                                            shape=(batch_size, hidden_size),
                                            usage=NodeFactory.ValueInfoType.input)

                nf_body.make_value_info(X_proj_subgraph,
                                        data_type=onnx.TensorProto.FLOAT,
                                        shape=(batch_size, 4*hidden_size),
                                        usage=NodeFactory.ValueInfoType.input)
                # subgraph nodes
                # it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
                # ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
                # ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
                # Ct = ft (.) Ct-1 + it (.) ct
                # ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
                # Ht = ot (.) h(Ct)
                prev_h_proj = nf_body.make_node('MatMul', [prev_h_subgraph, Rt])
                sum_x_proj_h_proj_bias = nf_body.make_node('Add', [X_proj_subgraph, prev_h_proj])
                split_outputs = ['split_i', 'split_o', 'split_f', 'split_c']
                nf_body.make_node('Split', sum_x_proj_h_proj_bias, {"axis":1, "split":[hidden_size]*4}, output_names=split_outputs)
                # manually add shape inference to split outputs
                for split_o in split_outputs:
                    nf_body.make_value_info(split_o,
                                            data_type=onnx.TensorProto.FLOAT,
                                            shape=(batch_size, hidden_size))
                activation_f, activation_g, activation_h = activations[direction_index*3:(direction_index+1)*3]
                it = nf_body.make_node(activation_f, 'split_i')
                ft = nf_body.make_node(activation_f, 'split_f')
                ct = nf_body.make_node(activation_g, 'split_c')
                c_subgraph = nf_body.make_node('Add',
                                               [nf_body.make_node('Mul', [ft, prev_c_subgraph]),
                                                nf_body.make_node('Mul', [it, ct])])
                ot = nf_body.make_node(activation_f, 'split_o')
                h_subgraph = nf_body.make_node('Mul', [ot, nf_body.make_node(activation_h, c_subgraph)])

                subgraph_outputs = handle_subgraph_outputs(nf_body,
                                                           seq_len_subgraph,
                                                           batch_size,
                                                           hidden_size,
                                                           [(h_subgraph, prev_h_subgraph),
                                                            (c_subgraph, prev_c_subgraph)] +
                                                           ([(h_subgraph, np.zeros(shape=(), dtype=np.float32))] if node.output[0] else [])) # skip scan output if node.output[0] is empty

                scan_attribs = {'body':scan_body,
                                'scan_input_directions':[is_backward],
                                'num_scan_inputs':1}
                if node.output[0]:
                    scan_attribs.update({'scan_output_directions':[is_backward]})
                scan = nf.make_node('Scan', ([seq_len] if seq_len else []) + [init_h, init_c, X_proj],
                                    scan_attribs,
                                    output_names=[o.name for o in subgraph_outputs[(0 if seq_len else 1):]])

                scan_h_outputs.append(subgraph_outputs[1])
                scan_c_outputs.append(subgraph_outputs[2])
                if node.output[0]:
                    scan_outputs.append(subgraph_outputs[3])

        handle_final_scan_outputs(node, nf, scan_outputs, [scan_h_outputs, scan_c_outputs], num_directions)

    # remove old initializers
    nf.remove_initializer(node.input[1])
    nf.remove_initializer(node.input[2])
    if num_inputs > 3:
        nf.remove_initializer(node.input[3])
    if num_inputs > 5:
        nf.remove_initializer(node.input[5], allow_empty=True)
    if num_inputs > 6:
        nf.remove_initializer(node.input[6], allow_empty=True)
    return True

def convert_gru_to_scan(node, out_main_graph):
    assert node.op_type == 'GRU'
    nf = NodeFactory(out_main_graph)
    with nf.scoped_prefix(node.output[0]) as scoped_prefix:
        X = node.input[0]
        Wa = nf.get_initializer(node.input[1])
        Ra = nf.get_initializer(node.input[2])
        num_inputs = len(node.input)
        Ba = nf.get_initializer(node.input[3]) if num_inputs > 3 else None
        seq_len = node.input[4] if num_inputs > 4 else None
        InitHa = node.input[5] if num_inputs > 5 else None

        direction, num_directions, activations = handle_common_attributes(node, ['Sigmoid', 'Tanh'])

        hidden_size = NodeFactory.get_attribute(node, 'hidden_size')
        linear_before_reset = NodeFactory.get_attribute(node, 'linear_before_reset')
        InitHa = handle_init_state(InitHa, nf, num_directions)

        batch_size, batch_node = handle_batch_size(X, nf, InitHa is None)
        if InitHa is None:
            zero_init_state = default_init_state(X, batch_size, batch_node, hidden_size, nf)

        scan_outputs = []
        scan_h_outputs = []
        for direction_index in range(num_directions):
            # for each direction
            # X [seq_len, batch_size, input_size]
            # W [3*hidden_size, input_size]
            # R [3*hidden_size, hidden_size]
            # B [6*hidden_size]
            # seq_len [batch_size]
            # init_h [batch_size, hidden_size]

            name_prefix = node.output[0] + '_' + str(direction_index) + '_'

            if InitHa is None:
                init_h = zero_init_state
            else:
                init_h = InitHa[direction_index]

            input_size = Wa.shape[len(Wa.shape) - 1]
            W_t = np.transpose(Wa[direction_index]) # [input_size, 3*hidden_size]
            R_t = np.transpose(Ra[direction_index]) # [hidden_size, 3*hidden_size]
            Rzr_t, Rh_t = np.hsplit(R_t, [2*hidden_size]) # [hidden_size, 2*hidden_size] and [hidden_size, hidden_size]
            Bzr, Bh = np.hsplit(Ba[direction_index].reshape(2, 3*hidden_size), [2*hidden_size])
            Bzr = Bzr.sum(axis=0) # [2*hidden_size]
            Wbh = Bh[0]
            Rbh = Bh[1]
            X_proj = nf.make_node('Add', [nf.make_node('MatMul', [X, W_t]), np.concatenate((Bzr, Wbh))]) #[seq_len, batch_size, 3*hidden_size]
            if num_directions == 1:
                is_backward = 0 if direction == 'forward' else 1
            else:
                is_backward = direction_index

            scan_body = onnx.GraphProto()
            scan_body.name = name_prefix + '_subgraph'

            nf_body = NodeFactory(out_main_graph, scan_body)
            with nf_body.scoped_prefix(name_prefix) as body_scoped_prefix:
                # subgraph inputs
                X_proj_subgraph = X_proj.name + '_subgraph'
                prev_h_subgraph = name_prefix + '_h_subgraph'

                seq_len_subgraph = declare_seq_len_in_subgraph(seq_len, nf_body, X_proj.name, batch_size)

                nf_body.make_value_info(prev_h_subgraph,
                                        data_type=onnx.TensorProto.FLOAT,
                                        shape=(batch_size, hidden_size),
                                        usage=NodeFactory.ValueInfoType.input)

                nf_body.make_value_info(X_proj_subgraph,
                                        data_type=onnx.TensorProto.FLOAT,
                                        shape=(batch_size, 3*hidden_size),
                                        usage=NodeFactory.ValueInfoType.input)

                # subgraph nodes
                # zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
                # rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
                # ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh) # default, when linear_before_reset = 0
                # ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh) # when linear_before_reset != 0
                # Ht = (1 - zt) (.) ht + zt (.) Ht-1

                split_X_outputs = ['split_Xzr', 'split_Xh']
                nf_body.make_node('Split', X_proj_subgraph, {"axis":1, "split":[2*hidden_size, hidden_size]}, output_names=split_X_outputs)
                nf_body.make_value_info('split_Xzr',
                                        data_type=onnx.TensorProto.FLOAT,
                                        shape=(batch_size, 2*hidden_size))
                nf_body.make_value_info('split_Xh',
                                        data_type=onnx.TensorProto.FLOAT,
                                        shape=(batch_size, hidden_size))

                activation_f, activation_g = activations[direction_index*2:(direction_index+1)*2]

                if linear_before_reset:
                    prev_h_proj = nf_body.make_node('Add', [nf_body.make_node('MatMul', [prev_h_subgraph, R_t]), np.concatenate((np.zeros(2*hidden_size).astype(np.float32), Rbh))])
                    split_prev_h_outputs = ['split_Hzr', 'split_Hh']
                    nf_body.make_node('Split', prev_h_proj, {"axis":1, "split":[2*hidden_size, hidden_size]}, output_names=split_prev_h_outputs)
                    nf_body.make_value_info('split_Hzr',
                                            data_type=onnx.TensorProto.FLOAT,
                                            shape=(batch_size, 2*hidden_size))
                    nf_body.make_value_info('split_Hh',
                                            data_type=onnx.TensorProto.FLOAT,
                                            shape=(batch_size, hidden_size))
                    ztrt = nf_body.make_node(activation_f, nf_body.make_node('Add', ['split_Hzr', 'split_Xzr']))
                    split_ztrt_outputs = ['split_zt', 'split_rt']
                    nf_body.make_node('Split', ztrt, {"axis":1, "split":[hidden_size, hidden_size]}, output_names=split_ztrt_outputs)
                    nf_body.make_value_info('split_zt',
                                            data_type=onnx.TensorProto.FLOAT,
                                            shape=(batch_size, hidden_size))
                    nf_body.make_value_info('split_rt',
                                            data_type=onnx.TensorProto.FLOAT,
                                            shape=(batch_size, hidden_size))
                    ht = nf_body.make_node(activation_g, nf_body.make_node('Add', [nf_body.make_node('Mul', ['split_rt', 'split_Hh']), 'split_Xh']))
                else:
                    ztrt = nf_body.make_node(activation_f, nf_body.make_node('Add', [nf_body.make_node('MatMul', [prev_h_subgraph, Rzr_t]), 'split_Xzr']))
                    split_ztrt_outputs = ['split_zt', 'split_rt']
                    nf_body.make_node('Split', ztrt, {"axis":1, "split":[hidden_size, hidden_size]}, output_names=split_ztrt_outputs)
                    nf_body.make_value_info('split_zt',
                                            data_type=onnx.TensorProto.FLOAT,
                                            shape=(batch_size, hidden_size))
                    nf_body.make_value_info('split_rt',
                                            data_type=onnx.TensorProto.FLOAT,
                                            shape=(batch_size, hidden_size))
                    ht = nf_body.make_node(activation_g, nf_body.make_node('Add', [nf_body.make_node('MatMul', [nf_body.make_node('Mul', [prev_h_subgraph, 'split_rt']), Rh_t]), 'split_Xh']))

                Ht = nf_body.make_node('Add', [nf_body.make_node('Mul', [nf_body.make_node('Sub', [np.asarray([1]).astype(np.float32),
                                                                                                   'split_zt']),
                                                                         ht]),
                                               nf_body.make_node('Mul', ['split_zt', prev_h_subgraph])])

                subgraph_outputs = handle_subgraph_outputs(nf_body,
                                                           seq_len_subgraph,
                                                           batch_size,
                                                           hidden_size,
                                                           [(Ht, prev_h_subgraph)] +
                                                           ([(Ht, np.zeros(shape=(), dtype=np.float32))] if node.output[0] else []))

                scan_attribs = {'body':scan_body,
                                'scan_input_directions':[is_backward],
                                'num_scan_inputs':1}
                if node.output[0]:
                    scan_attribs.update({'scan_output_directions':[is_backward]})
                scan = nf.make_node('Scan', ([seq_len] if seq_len else []) + [init_h, X_proj],
                                    scan_attribs,
                                    output_names=[o.name for o in subgraph_outputs[(0 if seq_len else 1):]])

                scan_h_outputs.append(subgraph_outputs[1])
                if node.output[0]:
                    scan_outputs.append(subgraph_outputs[2])

        handle_final_scan_outputs(node, nf, scan_outputs, [scan_h_outputs], num_directions)

    # remove old initializers
    nf.remove_initializer(node.input[1])
    nf.remove_initializer(node.input[2])
    if num_inputs > 3:
        nf.remove_initializer(node.input[3])
    if num_inputs > 5:
        nf.remove_initializer(node.input[5], allow_empty=True)
    return True

def convert_rnn_to_scan(node, out_main_graph):
    assert node.op_type == 'RNN'
    nf = NodeFactory(out_main_graph)
    with nf.scoped_prefix(node.output[0]) as scoped_prefix:
        X = node.input[0]
        Wa = nf.get_initializer(node.input[1])
        Ra = nf.get_initializer(node.input[2])
        num_inputs = len(node.input)
        Ba = nf.get_initializer(node.input[3]) if num_inputs > 3 else None
        seq_len = node.input[4] if num_inputs > 4 else None
        InitHa = node.input[5] if num_inputs > 5 else None

        direction, num_directions, activations = handle_common_attributes(node, ['Tanh'])

        hidden_size = NodeFactory.get_attribute(node, 'hidden_size')

        InitHa = handle_init_state(InitHa, nf, num_directions)

        batch_size, batch_node = handle_batch_size(X, nf, InitHa is None)
        if InitHa is None:
            zero_init_state = default_init_state(X, batch_size, batch_node, hidden_size, nf)

        scan_outputs = []
        scan_h_outputs = []
        for direction_index in range(num_directions):
            # for each direction
            # X [seq_len, batch_size, input_size]
            # W [hidden_size, input_size]
            # R [hidden_size, hidden_size]
            # B [2*hidden_size]
            # seq_len [batch_size]
            # init_h [batch_size, hidden_size]

            name_prefix = node.output[0] + '_' + str(direction_index) + '_'

            if InitHa is None:
                init_h = zero_init_state
            else:
                init_h = InitHa[direction_index]

            input_size = Wa.shape[len(Wa.shape) - 1]
            W_t = np.transpose(Wa[direction_index]) # [input_size, hidden_size]
            R_t = np.transpose(Ra[direction_index]) # [hidden_size, hidden_size]
            B = Ba[direction_index].reshape(2, hidden_size).sum(axis=0) # [hidden_size]
            X_proj = nf.make_node('Add', [nf.make_node('MatMul', [X, W_t]), B]) #[seq_len, batch_size, hidden_size]
            if num_directions == 1:
                is_backward = 0 if direction == 'forward' else 1
            else:
                is_backward = direction_index

            scan_body = onnx.GraphProto()
            scan_body.name = name_prefix + '_subgraph'

            nf_body = NodeFactory(out_main_graph, scan_body)
            with nf_body.scoped_prefix(name_prefix) as body_scoped_prefix:
                # subgraph inputs
                X_proj_subgraph = X_proj.name + '_subgraph'
                prev_h_subgraph = name_prefix + '_h_subgraph'

                seq_len_subgraph = declare_seq_len_in_subgraph(seq_len, nf_body, X_proj.name, batch_size)

                nf_body.make_value_info(prev_h_subgraph,
                                        data_type=onnx.TensorProto.FLOAT,
                                        shape=(batch_size, hidden_size),
                                        usage=NodeFactory.ValueInfoType.input)

                nf_body.make_value_info(X_proj_subgraph,
                                        data_type=onnx.TensorProto.FLOAT,
                                        shape=(batch_size, hidden_size),
                                        usage=NodeFactory.ValueInfoType.input)
                # subgraph nodes
                # Ht = f(Xt*(W^T) + Ht-1*(R^T) + Wb + Rb)

                activation_f = activations[direction_index]
                Ht = nf_body.make_node(activation_f, nf_body.make_node('Add', [nf_body.make_node('MatMul', [prev_h_subgraph, R_t]), X_proj_subgraph]))

                subgraph_outputs = handle_subgraph_outputs(nf_body,
                                                           seq_len_subgraph,
                                                           batch_size,
                                                           hidden_size,
                                                           [(Ht, prev_h_subgraph)] +
                                                           ([(Ht, np.zeros(shape=(), dtype=np.float32))] if node.output[0] else []))

                scan_attribs = {'body':scan_body,
                                'scan_input_directions':[is_backward],
                                'num_scan_inputs':1}
                if node.output[0]:
                    scan_attribs.update({'scan_output_directions':[is_backward]})
                scan = nf.make_node('Scan', ([seq_len] if seq_len else []) + [init_h, X_proj],
                                    scan_attribs,
                                    output_names=[o.name for o in subgraph_outputs[(0 if seq_len else 1):]])

                scan_h_outputs.append(subgraph_outputs[1])
                if node.output[0]:
                    scan_outputs.append(subgraph_outputs[2])

        handle_final_scan_outputs(node, nf, scan_outputs, [scan_h_outputs], num_directions)

    # remove old initializers
    nf.remove_initializer(node.input[1])
    nf.remove_initializer(node.input[2])
    if num_inputs > 3:
        nf.remove_initializer(node.input[3])
    if num_inputs > 5:
        nf.remove_initializer(node.input[5])
    return True

def convert_to_scan_model(input_model, output_model):
    in_mp = onnx.load(input_model)
    out_mp = onnx.ModelProto()
    out_mp.CopyFrom(in_mp)
    out_mp.ir_version = 5 # update ir version to avoid requirement of initializer in graph input
    ensure_opset(out_mp, 9) # bump up to ONNX opset 9, which is required for Scan
    out_mp.graph.ClearField('node')
    for in_n in in_mp.graph.node:
        if in_n.op_type in ['LSTM', 'GRU', 'RNN']:
            in_n = trim_unused_outputs(in_n, in_mp.graph)
        if in_n.op_type == 'LSTM':
            if convert_lstm_to_scan(in_n, out_mp.graph):
                continue
        if in_n.op_type == 'GRU':
            if convert_gru_to_scan(in_n, out_mp.graph):
                continue
        if in_n.op_type == 'RNN':
            if convert_rnn_to_scan(in_n, out_mp.graph):
                continue
        out_n = out_mp.graph.node.add()
        out_n.CopyFrom(in_n)

    onnx.save(out_mp, output_model)

def gemm_to_matmul(node, nf, converted_initializers):
    assert node.op_type == 'Gemm'

    alpha = NodeFactory.get_attribute(node, 'alpha', 1.0)
    beta = NodeFactory.get_attribute(node, 'beta', 1.0)
    transA = NodeFactory.get_attribute(node, 'transA', 0)
    transB = NodeFactory.get_attribute(node, 'transB', 0)

    A = node.input[0]
    B = node.input[1]
    Y = node.output[0]

    with nf.scoped_prefix(node.name) as scoped_prefix:
        if alpha != 1.0:
            alpha_name = node.name + '_Const_alpha'
            nf.make_initializer(np.full((), alpha, dtype=np.float32), alpha_name)
            alpha_A = nf.make_node('Mul', [alpha_name, A])
            A = alpha_A.name

        if transA:
            if A in converted_initializers:
                A = converted_initializers[A]
            else:
                A_initializer = nf.get_initializer(A)
                # A is an initializer
                if A_initializer is not None:
                    new_A = A + '_trans'
                    converted_initializers[A] = new_A
                    nf.make_initializer(np.transpose(A_initializer), new_A, in_main_graph=True)
                    nf.remove_initializer(A)
                    A = new_A
                else:
                    A = nf.make_node('Transpose', A)
        if transB:
            if B in converted_initializers:
                B = converted_initializers[B]
            else:
                B_initializer = nf.get_initializer(B)
                # B is an initializer
                if B_initializer is not None:
                    new_B = B + '_trans'
                    converted_initializers[B] = new_B
                    nf.make_initializer(np.transpose(B_initializer), new_B, in_main_graph=True)
                    nf.remove_initializer(B)
                    B = new_B
                else:
                    B = nf.make_node('Transpose', B)

        if len(node.input) != 3 or beta == 0.0:
            nf.make_node('MatMul', [A, B], output_names=Y)
        else:
            AB = nf.make_node('MatMul', [A, B])
            C = node.input[2]
            if beta != 1.0:
                beta_name = node.name + '_Const_beta'
                nf.make_initializer(np.full((), beta, dtype=np.float32), beta_name)
                C = nf.make_node('Mul', [beta_name, C])
            nf.make_node('Add', [AB, C], output_names=Y)

def convert_gemm_to_matmul(input_model, output_model):
    in_mp = onnx.load(input_model)
    out_mp = onnx.ModelProto()
    out_mp.CopyFrom(in_mp)
    out_mp.ir_version = 5 # update ir version to avoid requirement of initializer in graph input
    out_mp.graph.ClearField('node')
    nf = NodeFactory(out_mp.graph)
    # gemm_to_matmul will generate transposed weights if the corresponding input
    # comes from initializer. We keep a map between the original and converted
    # ones in case the original initializer is shared between Gemm ops
    converted_initializers = {}

    for in_n in in_mp.graph.node:
        if in_n.op_type == 'Gemm':
            gemm_to_matmul(in_n, nf, converted_initializers)
            continue

        out_n = out_mp.graph.node.add()
        out_n.CopyFrom(in_n)
        if in_n.op_type == 'Scan' or in_n.op_type == 'Loop':
            in_subgraph = NodeFactory.get_attribute(in_n, 'body')
            out_subgraph = NodeFactory.get_attribute(out_n, 'body')
            out_subgraph.ClearField('node')
            scan_nf = NodeFactory(out_mp.graph, out_subgraph)

            for in_sn in in_subgraph.node:
                if in_sn.op_type == 'Gemm':
                    gemm_to_matmul(in_sn, scan_nf, converted_initializers)
                    continue
                out_sn = out_subgraph.node.add()
                out_sn.CopyFrom(in_sn)

    onnx.save(out_mp, output_model)

# Old models (ir_version < 4) is required to initializers in graph inputs
# This is optional for ir_version >= 4
def remove_initializers_from_inputs(input_model, output_model, remain_inputs=[]):
    mp = onnx.load(input_model)

    def _append_initializer_from_graph(graph):
        initializers = [i.name for i in graph.initializer]
        for node in graph.node:
            if node.op_type == 'Scan': # currently only handle Scan
                subgraph = NodeFactory.get_attribute(node, 'body')
                initializers += _append_initializer_from_graph(subgraph)
        return initializers

    all_initializer_names = [n for n in _append_initializer_from_graph(mp.graph) if n not in remain_inputs]
    new_inputs = [vi for vi in mp.graph.input if not vi.name in all_initializer_names]
    mp.graph.ClearField('input')
    mp.graph.input.extend(new_inputs)
    onnx.save(mp, output_model)

def optimize_input_projection(input_model, output_model):
    in_mp = onnx.load(input_model)
    out_mp = onnx.ModelProto()
    out_mp.CopyFrom(in_mp)
    out_mp.ir_version = 5 # update ir version to avoid requirement of initializer in graph input
    out_mp.graph.ClearField('node')
    nf = NodeFactory(out_mp.graph, prefix='opt_inproj_')
    initializers = dict([(i.name, i) for i in in_mp.graph.initializer])
    # first find possible fused SVD and do constant folding on MatMul of initializers
    const_matmuls = [n for n in in_mp.graph.node if n.op_type == 'MatMul' and all([i in initializers for i in n.input])]
    for mm in const_matmuls:
        lhs = numpy_helper.to_array(initializers[mm.input[0]])
        rhs = numpy_helper.to_array(initializers[mm.input[1]])
        val = np.matmul(lhs, rhs)
        new_initializer = out_mp.graph.initializer.add()
        new_initializer.CopyFrom(numpy_helper.from_array(val, mm.output[0]))
        if not [n for n in in_mp.graph.node if n != mm and mm.input[0] in n.input]:
            nf.remove_initializer(mm.input[0])
        if not [n for n in in_mp.graph.node if n != mm and mm.input[1] in n.input]:
            nf.remove_initializer(mm.input[1])

    initializers = dict([(i.name,i) for i in out_mp.graph.initializer])

    # remove const_matmul output from graph outputs
    new_outputs = [i for i in out_mp.graph.output if not [m for m in const_matmuls if m.output[0] == i.name]]
    out_mp.graph.ClearField('output')
    out_mp.graph.output.extend(new_outputs)

    for in_n in in_mp.graph.node:
        if in_n in const_matmuls:
            continue

        optimize_scan = False
        if in_n.op_type == 'Scan':
            in_sg = NodeFactory.get_attribute(in_n, 'body')
            num_scan_inputs = NodeFactory.get_attribute(in_n, 'num_scan_inputs')
            # only support 1 scan input
            if num_scan_inputs == 1:
                optimize_scan = True

        # copy the node if it's not the scan node that is supported at the moment
        if not optimize_scan:
            out_n = out_mp.graph.node.add()
            out_n.CopyFrom(in_n)
            continue

        scan_input_directions = NodeFactory.get_attribute(in_n, 'scan_input_directions')
        scan_output_directions = NodeFactory.get_attribute(in_n, 'scan_output_directions')
        out_sg = onnx.GraphProto()
        out_sg.CopyFrom(in_sg)
        out_sg.ClearField('node')
        nf_subgraph = NodeFactory(out_mp.graph, out_sg, prefix='opt_inproj_sg_' + in_n.name + '_')
        new_inputs = list(in_n.input)
        in_sg_inputs = [i.name for i in in_sg.input]
        replaced_matmul = None
        for in_sn in in_sg.node:
            if in_sn.op_type == 'Concat' and len(in_sn.input) == 2 and all([i in in_sg_inputs for i in in_sn.input]):
                # make sure the concat's inputs are scan input and scan state
                if NodeFactory.get_attribute(in_sn, 'axis') != len(in_sg.input[-1].type.tensor_type.shape.dim) - 1:
                    continue # must concat last dim
                matmul_node = [nn for nn in in_sg.node if nn.op_type == 'MatMul' and in_sn.output[0] in nn.input]
                if not matmul_node:
                    continue
                replaced_matmul = matmul_node[0]
                assert replaced_matmul.input[1] in initializers
                aa = nf.get_initializer(replaced_matmul.input[1])
                input_size = in_sg.input[-1].type.tensor_type.shape.dim[-1].dim_value
                if in_sg_inputs[-1] == in_sn.input[0]:
                    hidden_idx = 1
                    input_proj_weights, hidden_proj_weights = np.vsplit(aa, [input_size])
                else:
                    hidden_idx = 0
                    hidden_proj_weights, input_proj_weights = np.vsplit(aa, [aa.shape[-1] - input_size])
                # add matmul for input_proj outside of Scan
                input_proj = nf.make_node('MatMul', [new_inputs[-1], input_proj_weights])
                input_proj.doc_string = replaced_matmul.doc_string
                new_inputs[-1] = input_proj.name
                out_sg.input[-1].type.tensor_type.shape.dim[-1].dim_value = input_proj_weights.shape[-1]
                # add matmul for hidden_proj inside Scan
                hidden_proj = nf_subgraph.make_node('MatMul', [in_sn.input[hidden_idx], hidden_proj_weights])
                hidden_proj.doc_string = replaced_matmul.doc_string
                nf_subgraph.make_node('Add', [out_sg.input[-1].name, hidden_proj], output_names=replaced_matmul.output[0])
                # remove initializer of concat matmul
                if not [n for n in in_mp.graph.node if n != in_n and replaced_matmul.input[1] in n.input]:
                    nf.remove_initializer(replaced_matmul.input[1])
            elif in_sn != replaced_matmul:
                out_sg.node.add().CopyFrom(in_sn)

        scan = nf.make_node('Scan', new_inputs,
                            {'body':out_sg,
                              'scan_input_directions':scan_input_directions,
                              'scan_output_directions':scan_output_directions,
                              'num_scan_inputs':num_scan_inputs},
                            output_names=list(in_n.output))
        scan.name = in_n.name
        scan.doc_string = in_n.doc_string

    onnx.save(out_mp, output_model)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help='The modification mode',
                        choices=['to_scan',
                                 'opt_inproj',
                                 'gemm_to_matmul',
                                 'remove_initializers_from_inputs'])
    parser.add_argument('--input', help='The input model file', default=None)
    parser.add_argument('--output', help='The output model file', default=None)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    print('input model: ' + args.input)
    print('output model ' + args.output)
    if args.mode == 'to_scan':
        print('Convert LSTM/GRU/RNN to Scan...')
        convert_to_scan_model(args.input, args.output)
    elif args.mode == 'gemm_to_matmul':
        print('Convert Gemm to MatMul')
        convert_gemm_to_matmul(args.input, args.output)
    elif args.mode == 'opt_inproj':
        print('Optimize input projection in Scan...')
        optimize_input_projection(args.input, args.output)
    elif args.mode == 'remove_initializers_from_inputs':
        print('Remove all initializers from input for model with IR version >= 4...')
        remove_initializers_from_inputs(args.input, args.output)
    else:
        raise NotImplementedError('Unknown mode')
    print('Running symbolic shape inference on output model')
    SymbolicShapeInference.infer_shapes(args.output, args.output, auto_merge=True)
    print('Done!')

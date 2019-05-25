import argparse
import onnx
import struct
import numpy as np
from onnx import helper, numpy_helper
from enum import Enum
import re

class QuantizeConfig:
    def __init__(self, signed, reserved_bits, type_bits=8):
        self.sign_bit_ = 1 if signed else 0
        self.reserved_bits_ = reserved_bits
        self.type_bits_ = type_bits

    def signed(self):
        return self.sign_bit_ == 1

    def usable_bits(self):
        return self.type_bits_ - self.reserved_bits_

    def q_max(self):
        return float((1 << (self.usable_bits() - self.sign_bit_)) - 1)

    def q_min(self):
        return float(-(self.q_max() + 1) if self.signed()  else 0)

    def q_span(self):
        return float(1 << self.usable_bits()) - 1

    def q_range(self):
        return self.q_max() + 0.5 if self.signed() else float(1 << self.usable_bits())

class NodeFactory:
    node_count_ = 0
    const_count_ = 0

    def __init__(self, main_graph, sub_graph=None):
        self.graph_ = sub_graph if sub_graph else main_graph
        self.main_graph_ = main_graph
        self.name_prefix_ = ''

    def set_prefix(self, prefix):
        self.name_prefix_ = prefix

    def get_initializer(self, name):
        found = [i for i in self.main_graph_.initializer if i.name == name]
        if found:
            return numpy_helper.to_array(found[0])
        else:
            return None

    def remove_initializer(self, name):
        initializer = [i for i in self.main_graph_.initializer if i.name == name]
        assert initializer
        self.main_graph_.initializer.remove(initializer[0])
        initializer_in_input = [i for i in self.main_graph_.input if i.name == name]
        if initializer_in_input:
            self.main_graph_.input.remove(initializer_in_input[0])

    @staticmethod
    def get_attribute(node, attr_name, default_value=None):
        found = [attr for attr in node.attribute if attr.name == attr_name]
        if found:
            return helper.get_attribute_value(found[0])
        else:
            return default_value

    class ValueInfoType(Enum):
        input = 1
        output = 2
        initializer = 3

    def make_value_info(self, node_or_name, data_type, shape=None, usage=None):
        if usage == NodeFactory.ValueInfoType.input:
            value_info = self.graph_.input.add()
        elif usage == NodeFactory.ValueInfoType.output:
            value_info = self.graph_.output.add()
        elif usage == NodeFactory.ValueInfoType.initializer:
            # initializer always stay in main_graph as input
            value_info = self.main_graph_.input.add()
        elif not usage:
            value_info = self.graph_.value_info.add()
        else:
            raise NotImplementedError("unknown usage")

        if type(node_or_name) == str:
            name = node_or_name
        else:
            assert len(node_or_name.output) == 1
            name = node_or_name.output[0]

        value_info.CopyFrom(helper.make_tensor_value_info(name, data_type, shape))

    def make_initializer(self, ndarray, name='', add_value_info=True):
        # initializers are stored only in main graph
        new_initializer = self.main_graph_.initializer.add()
        new_name = name
        if len(new_name) == 0:
            new_name = self.name_prefix_ + '_Const_' + str(NodeFactory.const_count_)
            NodeFactory.const_count_ = NodeFactory.const_count_ + 1
        new_initializer.CopyFrom(numpy_helper.from_array(ndarray, new_name))
        if add_value_info:
            self.make_value_info(new_initializer.name, new_initializer.data_type, ndarray.shape, usage=NodeFactory.ValueInfoType.initializer)
        return new_initializer

    def make_node(self, op_type, inputs, attributes={}, output_names=None, node=None):
        if type(inputs) != list:
            inputs = [inputs]
        if output_names and type(output_names) != list:
            output_names = [output_names]
        input_names = []
        for i in inputs:
            if type(i) == onnx.NodeProto:
                input_names.append(i.name)
            elif type(i) == str:
                input_names.append(i)
            elif type(i) == np.ndarray:
                new_initializer = self.make_initializer(i)
                input_names.append(new_initializer.name)

        if not node:
            node = self.graph_.node.add()

        name = self.name_prefix_ + op_type + '_' + str(NodeFactory.node_count_)
        NodeFactory.node_count_ = NodeFactory.node_count_ + 1

        if not output_names:
            output_names = [name]

        node.CopyFrom(helper.make_node(op_type, input_names, output_names, name, **attributes))
        return node

def quantize_matmul_2d_with_weight(in_node, in_graph, nf, converted_weights):
    assert in_node.op_type == 'MatMul'

    # quantize weight
    # only handles weight being inputs[1] of MatMul/Gemm node
    fparam_name = in_node.input[1]

    # skip if weights shared by other nodes that's not MatMul
    # TODO: support GEMM if needed
    other_nodes = [n for n in in_graph.node if n != in_node and fparam_name in n.input and n.op_type != 'MatMul']
    if other_nodes:
        return False

    if fparam_name in converted_weights:
        step, base, qparam_rowsum, qparam = converted_weights[fparam_name]
    else:
        fparam = nf.get_initializer(fparam_name)
        if fparam is None or len(fparam.shape) != 2:
            return False

        fmin = np.amin(fparam, axis=0)
        fmax = np.amax(fparam, axis=0)
        w_qcfg = QuantizeConfig(reserved_bits=0, signed=1) # quantize parameter to signed int, with no reserved bit
        span = fmax - fmin;
        q_range = w_qcfg.q_span() # use w_qcfg.q_range() leads to 1 more reserved bit
        step = span / q_range
        base = (fmax + fmin + step) * 0.5 if w_qcfg.signed() else fmin
        fparam_norm = np.zeros_like(fparam)
        expand_span = np.expand_dims(span,0)
        np.divide((fparam - np.expand_dims(base,0)), expand_span, out=fparam_norm, where=expand_span!=0)
        qparam = np.round(fparam_norm * q_range)
        qparam = np.clip(qparam, w_qcfg.q_min(), w_qcfg.q_max())
        qparam_rowsum = np.sum(qparam, axis=0)
        qparam = qparam.astype(np.int8)
        nf.make_initializer(step, fparam_name + '_step')
        nf.make_initializer(base, fparam_name + '_base')
        nf.make_initializer(qparam_rowsum, fparam_name + '_qparam_rowsum')
        nf.make_initializer(qparam, fparam_name + '_qparam')
        step = fparam_name + '_step'
        base = fparam_name + '_base'
        qparam_rowsum = fparam_name + '_qparam_rowsum'
        qparam = fparam_name + '_qparam'
        converted_weights[fparam_name] = (step, base, qparam_rowsum, qparam)
        nf.remove_initializer(fparam_name)

    # quantize input
    nf.set_prefix(in_node.name)
    input_dim = nf.get_initializer(qparam).shape[0]
    x_qcfg = QuantizeConfig(reserved_bits=1, signed=0) # quantize input to unsigned int, with 1 reserved bit
    # Add quantization for X
    X = in_node.input[0]
    reduce_max_X = nf.make_node('ReduceMax', X, {'axes':[-1]}) # keepdims = 1
    bias_X = nf.make_node('ReduceMin', X, {'axes':[-1]})
    delta_X = nf.make_node('Sub', [reduce_max_X, bias_X])
    scale_X = nf.make_node('Div', [delta_X, np.asarray(x_qcfg.q_range()).astype(np.float32)])
    norm_X = nf.make_node('Div', [nf.make_node('Sub', [X, bias_X]), delta_X])
    Q_Xf = nf.make_node('Mul', [norm_X, np.asarray(x_qcfg.q_range()).astype(np.float32)])
    Q_Xf = nf.make_node('Add', [Q_Xf, np.asarray(0.5).astype(np.float32)])
    Q_Xf = nf.make_node('Floor', Q_Xf)
    Q_Xf = nf.make_node('Clip', Q_Xf, {'max':x_qcfg.q_max(), 'min':x_qcfg.q_min()})
    Q_X = nf.make_node('Cast', Q_Xf, {'to':int(onnx.TensorProto.UINT8)})
    Q_X_sum = nf.make_node('ReduceSum', Q_Xf, {'axes':[-1]})

    # MatMulInteger
    Q_Y = nf.make_node('MatMulInteger', [Q_X, qparam])
    nf.make_value_info(Q_Y, data_type=onnx.TensorProto.INT32)

    # Dequantize
    Y = in_node.output[0]
    o0 = nf.make_node('Mul', [nf.make_node('Mul', [step, scale_X]),
                              nf.make_node('Cast', Q_Y, {'to': int(onnx.TensorProto.FLOAT)})])
    o1 = nf.make_node('Mul', [nf.make_node('Mul', [step, bias_X]), qparam_rowsum])
    o2 = nf.make_node('Mul', [base, nf.make_node('Mul', [scale_X, Q_X_sum])])
    o3 = nf.make_node('Mul', [base, nf.make_node('Mul', [bias_X, np.asarray(float(input_dim)).astype(np.float32)])])

    nf.make_node('Add', [nf.make_node('Add', [nf.make_node('Add', [o3, o2]), o1]), o0], output_names=Y)
    return True

def convert_lstm_to_scan(node, out_main_graph):
    assert node.op_type == 'LSTM'
    nf = NodeFactory(out_main_graph)
    nf.set_prefix(node.name)

    X = node.input[0]
    Wa = nf.get_initializer(node.input[1])
    Ra = nf.get_initializer(node.input[2])
    num_inputs = len(node.input)
    Ba = nf.get_initializer(node.input[3]) if num_inputs > 3 else None
    seq_len = node.input[4] if num_inputs > 4 else None
    InitHa = node.input[5] if num_inputs > 5 else None
    InitCa = node.input[6] if num_inputs > 6 else None
    PB = node.input[7] if num_inputs > 7 else None

    # TODO: support seq_len
    assert not seq_len
    # TODO: support peephole
    assert not PB
    # TODO: support Y_h/Y_c
    assert len(node.output) == 1

    direction = NodeFactory.get_attribute(node, 'direction')
    if direction:
        direction = str(direction, 'utf-8')
    else:
        direction = 'forward'
    num_directions = 2 if direction == 'bidirectional' else 1

    activations = NodeFactory.get_attribute(node, 'activations')
    if activations:
        activations = [str(x, 'utf-8') for x in activations]
    else:
        activations = ['Sigmoid', 'Tanh', 'Tanh'] * num_directions

    activation_alpha = NodeFactory.get_attribute(node, 'activation_alpha')
    activation_beta = NodeFactory.get_attribute(node, 'activation_beta')
    clip_threshold = NodeFactory.get_attribute(node, 'clip')
    # TODO: support these activation attributes
    assert not activation_alpha
    assert not activation_beta
    assert not clip_threshold

    hidden_size = NodeFactory.get_attribute(node, 'hidden_size')
    input_forget = NodeFactory.get_attribute(node, 'input_forget')

    # TODO: implement input_forget = 1
    assert not (input_forget != None and input_forget == 1)

    # TODO: support symbolic batch size
    batch_size = 1

    scan_outputs = []
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

        name_prefix = node.name + '_' + str(direction_index) + '_'

        if not InitHa:
            init_h = np.zeros(shape=(batch_size, hidden_size), dtype=np.float32)
        else:
            init_h = nf.get_initializer(InitHa)[direction_index]

        if not InitCa:
            init_c = np.zeros(shape=(batch_size, hidden_size), dtype=np.float32)
        else:
            init_c = nf.get_initializer(InitCa)[direction_index]

        input_size = Wa.shape[len(Wa.shape) - 1]
        Wt = np.transpose(Wa[direction_index])
        Rt = np.transpose(Ra[direction_index])
        B = Ba[direction_index].reshape(2, 4*hidden_size).sum(axis=0) # [4*hidden_size]
        X_proj = nf.make_node('MatMul', [X, Wt]) #[seq_len, batch_size, 4*hidden_size]
        if num_directions == 1:
            is_backward = 0 if direction == 'forward' else 1
        else:
            is_backward = direction_index

        scan_body = onnx.GraphProto()
        scan_body.name = name_prefix + '_subgraph'
        scan_output = name_prefix + '_Output'
        scan_h_output = name_prefix + '_h'
        scan_c_output = name_prefix + '_c'

        nf_body = NodeFactory(out_main_graph, scan_body)

        # subgraph inputs
        X_proj_subgraph = X_proj.name + '_subgraph'
        prev_h_subgraph = name_prefix + '_h_subgraph'
        prev_c_subgraph = name_prefix + '_c_subgraph'

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
        sum_x_proj_h_proj_bias = nf_body.make_node('Add', [X_proj_subgraph, nf_body.make_node('Add', [prev_h_proj, B])])
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
        # hook subgraph output
        subgraph_output = nf_body.make_node('Identity', h_subgraph)
        nf_body.make_node('Identity', h_subgraph, output_names=scan_h_output)
        nf_body.make_node('Identity', c_subgraph, output_names=scan_c_output)

        for subgraph_o in [scan_h_output, scan_c_output, subgraph_output]:
            nf_body.make_value_info(subgraph_o,
                                    data_type=onnx.TensorProto.FLOAT,
                                    shape=(batch_size, hidden_size),
                                    usage=NodeFactory.ValueInfoType.output)

        scan = nf.make_node('Scan', [init_h, init_c, X_proj],
                            {'body':scan_body,
                              'scan_input_directions':[is_backward],
                              'scan_output_directions':[is_backward],
                              'num_scan_inputs':1},
                            output_names=[scan_h_output, scan_c_output, scan_output])

        scan_outputs.append(scan_output)

    if num_directions == 2:
        scan_outputs = [nf.make_node('Unsqueeze', x, {'axes':[1]}) for x in scan_outputs]
        nf.make_node('Concat', scan_outputs, {'axis':1}, output_names=node.output[0])
    else:
        nf.make_node('Unsqueeze', scan_outputs[0], {'axes':[1]}, output_names=node.output[0])

    # remove old initializers
    nf.remove_initializer(node.input[1])
    nf.remove_initializer(node.input[2])
    if num_inputs > 3:
        nf.remove_initializer(node.input[3])
    if num_inputs > 5:
        nf.remove_initializer(node.input[5])
    if num_inputs > 6:
        nf.remove_initializer(node.input[6])
    return True

def convert_gru_to_scan(node, out_main_graph):
    assert node.op_type == 'GRU'
    nf = NodeFactory(out_main_graph)
    nf.set_prefix(node.name)

    X = node.input[0]
    Wa = nf.get_initializer(node.input[1])
    Ra = nf.get_initializer(node.input[2])
    num_inputs = len(node.input)
    Ba = nf.get_initializer(node.input[3]) if num_inputs > 3 else None
    seq_len = node.input[4] if num_inputs > 4 else None
    InitHa = node.input[5] if num_inputs > 5 else None

    # TODO: support seq_len
    assert not seq_len
    # TODO: support Y_h
    assert len(node.output) == 1

    direction = NodeFactory.get_attribute(node, 'direction')
    if direction:
        direction = str(direction, 'utf-8')
    else:
        direction = 'forward'
    num_directions = 2 if direction == 'bidirectional' else 1

    activations = NodeFactory.get_attribute(node, 'activations')
    if activations:
        activations = [str(x, 'utf-8') for x in activations]
    else:
        activations = ['Sigmoid', 'Tanh'] * num_directions

    activation_alpha = NodeFactory.get_attribute(node, 'activation_alpha')
    activation_beta = NodeFactory.get_attribute(node, 'activation_beta')
    clip_threshold = NodeFactory.get_attribute(node, 'clip')
    # TODO: support these activation attributes
    assert not activation_alpha
    assert not activation_beta
    assert not clip_threshold

    hidden_size = NodeFactory.get_attribute(node, 'hidden_size')
    linear_before_reset = NodeFactory.get_attribute(node, 'linear_before_reset')

    # TODO: support symbolic batch size
    batch_size = 1

    scan_outputs = []
    for direction_index in range(num_directions):
        # for each direction
        # X [seq_len, batch_size, input_size]
        # W [3*hidden_size, input_size]
        # R [3*hidden_size, hidden_size]
        # B [6*hidden_size]
        # seq_len [batch_size]
        # init_h [batch_size, hidden_size]

        name_prefix = node.name + '_' + str(direction_index) + '_'

        if not InitHa:
            init_h = np.zeros(shape=(batch_size, hidden_size), dtype=np.float32)
        else:
            init_h = nf.get_initializer(InitHa)[direction_index]

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
        scan_output = name_prefix + '_Output'
        scan_h_output = name_prefix + '_h'

        nf_body = NodeFactory(out_main_graph, scan_body)

        # subgraph inputs
        X_proj_subgraph = X_proj.name + '_subgraph'
        prev_h_subgraph = name_prefix + '_h_subgraph'

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

        # hook subgraph output
        subgraph_output = nf_body.make_node('Identity', Ht)
        nf_body.make_node('Identity', Ht, output_names=scan_h_output)

        for subgraph_o in [scan_h_output, subgraph_output]:
            nf_body.make_value_info(subgraph_o,
                                    data_type=onnx.TensorProto.FLOAT,
                                    shape=(batch_size, hidden_size),
                                    usage=NodeFactory.ValueInfoType.output)

        scan = nf.make_node('Scan', [init_h, X_proj],
                            {'body':scan_body,
                              'scan_input_directions':[is_backward],
                              'scan_output_directions':[is_backward],
                              'num_scan_inputs':1},
                            output_names=[scan_h_output, scan_output])

        scan_outputs.append(scan_output)

    if num_directions == 2:
        scan_outputs = [nf.make_node('Unsqueeze', x, {'axes':[1]}) for x in scan_outputs]
        nf.make_node('Concat', scan_outputs, {'axis':1}, output_names=node.output[0])
    else:
        nf.make_node('Unsqueeze', scan_outputs[0], {'axes':[1]}, output_names=node.output[0])

    # remove old initializers
    nf.remove_initializer(node.input[1])
    nf.remove_initializer(node.input[2])
    if num_inputs > 3:
        nf.remove_initializer(node.input[3])
    if num_inputs > 5:
        nf.remove_initializer(node.input[5])
    return True

def convert_rnn_to_scan(node, out_main_graph):
    assert node.op_type == 'RNN'
    nf = NodeFactory(out_main_graph)
    nf.set_prefix(node.name)

    X = node.input[0]
    Wa = nf.get_initializer(node.input[1])
    Ra = nf.get_initializer(node.input[2])
    num_inputs = len(node.input)
    Ba = nf.get_initializer(node.input[3]) if num_inputs > 3 else None
    seq_len = node.input[4] if num_inputs > 4 else None
    InitHa = node.input[5] if num_inputs > 5 else None

    # TODO: support seq_len
    assert not seq_len
    # TODO: support Y_h
    assert len(node.output) == 1

    direction = NodeFactory.get_attribute(node, 'direction')
    if direction:
        direction = str(direction, 'utf-8')
    else:
        direction = 'forward'
    num_directions = 2 if direction == 'bidirectional' else 1

    activations = NodeFactory.get_attribute(node, 'activations')
    if activations:
        activations = [str(x, 'utf-8') for x in activations]
    else:
        activations = ['Tanh'] * num_directions

    activation_alpha = NodeFactory.get_attribute(node, 'activation_alpha')
    activation_beta = NodeFactory.get_attribute(node, 'activation_beta')
    clip_threshold = NodeFactory.get_attribute(node, 'clip')
    # TODO: support these activation attributes
    assert not activation_alpha
    assert not activation_beta
    assert not clip_threshold

    hidden_size = NodeFactory.get_attribute(node, 'hidden_size')

    # TODO: support symbolic batch size
    batch_size = 1

    scan_outputs = []
    for direction_index in range(num_directions):
        # for each direction
        # X [seq_len, batch_size, input_size]
        # W [hidden_size, input_size]
        # R [hidden_size, hidden_size]
        # B [2*hidden_size]
        # seq_len [batch_size]
        # init_h [batch_size, hidden_size]

        name_prefix = node.name + '_' + str(direction_index) + '_'

        if not InitHa:
            init_h = np.zeros(shape=(batch_size, hidden_size), dtype=np.float32)
        else:
            init_h = nf.get_initializer(InitHa)[direction_index]

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
        scan_output = name_prefix + '_Output'
        scan_h_output = name_prefix + '_h'

        nf_body = NodeFactory(out_main_graph, scan_body)

        # subgraph inputs
        X_proj_subgraph = X_proj.name + '_subgraph'
        prev_h_subgraph = name_prefix + '_h_subgraph'

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

        # hook subgraph output
        subgraph_output = nf_body.make_node('Identity', Ht)
        nf_body.make_node('Identity', Ht, output_names=scan_h_output)

        for subgraph_o in [scan_h_output, subgraph_output]:
            nf_body.make_value_info(subgraph_o,
                                    data_type=onnx.TensorProto.FLOAT,
                                    shape=(batch_size, hidden_size),
                                    usage=NodeFactory.ValueInfoType.output)

        scan = nf.make_node('Scan', [init_h, X_proj],
                            {'body':scan_body,
                              'scan_input_directions':[is_backward],
                              'scan_output_directions':[is_backward],
                              'num_scan_inputs':1},
                            output_names=[scan_h_output, scan_output])

        scan_outputs.append(scan_output)

    if num_directions == 2:
        scan_outputs = [nf.make_node('Unsqueeze', x, {'axes':[1]}) for x in scan_outputs]
        nf.make_node('Concat', scan_outputs, {'axis':1}, output_names=node.output[0])
    else:
        nf.make_node('Unsqueeze', scan_outputs[0], {'axes':[1]}, output_names=node.output[0])

    # remove old initializers
    nf.remove_initializer(node.input[1])
    nf.remove_initializer(node.input[2])
    if num_inputs > 3:
        nf.remove_initializer(node.input[3])
    if num_inputs > 5:
        nf.remove_initializer(node.input[5])
    return True

# quantize matmul to MatMulInteger using asymm uint8
def convert_matmul_model(input_model, output_model, only_for_scan):
    in_mp = onnx.load(input_model)
    out_mp = onnx.ModelProto()
    out_mp.CopyFrom(in_mp)
    for opset in out_mp.opset_import:
        if opset.domain == '' or opset.domain == 'onnx':
            opset.version = 10 # bump up to opset 10, which is required for MatMulInteger
    out_mp.graph.ClearField('node')

    nf = NodeFactory(out_mp.graph)
    converted_weights = {} # remember MatMul weights that have been converted, in case of sharing
    for in_n in in_mp.graph.node:
        # convert opset9 Slice to opset10
        if in_n.op_type == 'Slice' and len(in_n.input) == 1:
            nf.set_prefix(in_n.name)
            slice_inputs = [in_n.input[0],
                            np.asarray(NodeFactory.get_attribute(in_n,'starts')).astype(np.int32),
                            np.asarray(NodeFactory.get_attribute(in_n,'ends')).astype(np.int32),
                            np.asarray(NodeFactory.get_attribute(in_n,'axes')).astype(np.int32)]
            nf.make_node('Slice', slice_inputs, output_names=[in_n.output[0]])
            continue

        if in_n.op_type == 'MatMul' and not only_for_scan:
            if quantize_matmul_2d_with_weight(in_n, in_mp.graph, nf, converted_weights):
                continue

        out_n = out_mp.graph.node.add()
        out_n.CopyFrom(in_n)
        if in_n.op_type == 'Scan':
            in_body = [attr for attr in in_n.attribute if attr.name == 'body'][0]
            out_body = [attr for attr in out_n.attribute if attr.name == 'body'][0]
            out_body.g.ClearField('node')
            scan_nf = NodeFactory(out_mp.graph, out_body.g)

            # for Scan using Slice, try change to split to avoid opset10 Slice with dynamic output shape
            subgraph_slice_nodes = [in_sn for in_sn in in_body.g.node if in_sn.op_type == 'Slice']
            subgraph_slice_map = {}
            subgraph_slice_axis = 0
            subgraph_split_node = None
            if subgraph_slice_nodes:
                subgraph_slice_inputs = set([in_sn.input[0] for in_sn in subgraph_slice_nodes])
                # try change to split only when all slices have the same input
                if len(subgraph_slice_inputs) == 1:
                    subgraph_slice_starts = [NodeFactory.get_attribute(in_n,'starts') for in_n in subgraph_slice_nodes]
                    subgraph_slice_ends = [NodeFactory.get_attribute(in_n,'ends') for in_n in subgraph_slice_nodes]
                    subgraph_slice_axes = [NodeFactory.get_attribute(in_n,'axes')  for in_n in subgraph_slice_nodes]
                    if len(subgraph_slice_axes[0]) == 1:
                        aa = np.asarray(subgraph_slice_axes)
                        if np.min(aa) == np.max(aa):
                            subgraph_slice_axis = np.min(aa)
                            start = np.sort(np.asarray(subgraph_slice_starts).reshape(-1))
                            end = np.sort(np.asarray(subgraph_slice_ends).reshape(-1))
                            if all(start[1:] == end[:-1]) and start[0] == 0:
                                for s in start:
                                    split_index = subgraph_slice_starts.index([s])
                                    subgraph_slice_map[subgraph_slice_nodes[split_index].output[0]] = subgraph_slice_ends[split_index][0] - subgraph_slice_starts[split_index][0]

            for in_sn in in_body.g.node:
                if in_sn.op_type == 'MatMul':
                    if quantize_matmul_2d_with_weight(in_sn, in_body.g, scan_nf, converted_weights):
                        continue

                if in_sn.op_type == 'Slice' and len(in_sn.input) == 1:
                    if subgraph_slice_map:
                        if not subgraph_split_node:
                            subgraph_split_node = scan_nf.make_node('Split',
                                                                    in_sn.input[0],
                                                                    {'axis':subgraph_slice_axis, 'split':[v for v in subgraph_slice_map.values()]},
                                                                    output_names=[k for k in subgraph_slice_map.keys()])
                        continue
                    else:
                        scan_nf.set_prefix(in_sn.name)
                        slice_inputs = [in_sn.input[0],
                                        np.asarray(NodeFactory.get_attribute(in_sn,'starts')).astype(np.int32),
                                        np.asarray(NodeFactory.get_attribute(in_sn,'ends')).astype(np.int32),
                                        np.asarray(NodeFactory.get_attribute(in_sn,'axes')).astype(np.int32)]
                        scan_nf.make_node('Slice', slice_inputs, output_names=[in_sn.output[0]])
                        continue

                out_sn = out_body.g.node.add()
                out_sn.CopyFrom(in_sn)

    onnx.save(out_mp, output_model)

def convert_to_scan_model(input_model, output_model):
    in_mp = onnx.load(input_model)
    out_mp = onnx.ModelProto()
    out_mp.CopyFrom(in_mp)
    out_mp.graph.ClearField('node')
    for in_n in in_mp.graph.node:
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

def optimize_input_projection(input_model, output_model):
    in_mp = onnx.load(input_model)
    out_mp = onnx.ModelProto()
    out_mp.CopyFrom(in_mp)
    out_mp.graph.ClearField('node')
    
    nf = NodeFactory(out_mp.graph)
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

        if in_n.op_type == 'Scan':
            in_sg = NodeFactory.get_attribute(in_n, 'body')
            num_scan_inputs = NodeFactory.get_attribute(in_n, 'num_scan_inputs')
            scan_input_directions = NodeFactory.get_attribute(in_n, 'scan_input_directions')
            scan_output_directions = NodeFactory.get_attribute(in_n, 'scan_output_directions')
            out_sg = onnx.GraphProto()
            out_sg.CopyFrom(in_sg)
            out_sg.ClearField('node')
            nf_subgraph = NodeFactory(out_mp.graph, out_sg)
            # only support 1 scan input
            assert num_scan_inputs == 1
            new_inputs = list(in_n.input)
            in_sg_inputs = [i.name for i in in_sg.input]
            replaced_matmul = None
            for in_sn in in_sg.node:
                if in_sn.op_type == 'Concat' and all([i in in_sg_inputs for i in in_sn.input]):
                    # make sure the concat's inputs are scan input and scan state
                    assert NodeFactory.get_attribute(in_sn, 'axis') == 1
                    assert in_sn.input[0] == in_sg_inputs[-1]
                    matmul_node = [nn for nn in in_sg.node if nn.op_type == 'MatMul' and in_sn.output[0] in nn.input]
                    assert matmul_node
                    replaced_matmul = matmul_node[0]
                    assert replaced_matmul.input[1] in initializers
                    aa = nf.get_initializer(replaced_matmul.input[1])
                    input_size = in_sg.input[-1].type.tensor_type.shape.dim[-1].dim_value
                    input_proj_weights, hidden_proj_weights = np.vsplit(aa, [input_size])
                    # add matmul for input_proj outside of Scan
                    input_proj = nf.make_node('MatMul', [new_inputs[-1], input_proj_weights])
                    new_inputs[-1] = input_proj.name
                    out_sg.input[-1].type.tensor_type.shape.dim[-1].dim_value = input_proj_weights.shape[-1]
                    # add matmul for hidden_proj inside Scan
                    hidden_proj = nf_subgraph.make_node('MatMul', [in_sn.input[1], hidden_proj_weights])
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
        else:
            out_n = out_mp.graph.node.add()
            out_n.CopyFrom(in_n)

    onnx.save(out_mp, output_model)

def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('--mode', help='The modification mode', choices=['to_scan', 'to_imatmul', 'opt_inproj'])
  parser.add_argument('--input', help='The input model file', default=None)
  parser.add_argument('--output', help='The input model file', default=None)
  parser.add_argument('--only_for_scan', help='Apply editing only inside scan', action='store_true', default=False)
  return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    print('input model: ' + args.input)
    print('output model ' + args.output)
    if args.mode == 'to_imatmul':
        print('Quantize MatMul to MatMulInteger...')
        convert_matmul_model(args.input, args.output, args.only_for_scan)
    elif args.mode == 'to_scan':
        print('Convert LSTM to Scan...')
        convert_to_scan_model(args.input, args.output)
    elif args.mode == 'opt_inproj':
        print('Optimize input projection in Scan...')
        optimize_input_projection(args.input, args.output)
    else:
        raise NotImplementedError('Unknown mode')
    print('Done!')
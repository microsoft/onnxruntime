# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# -*- coding: UTF-8 -*-
import argparse
from enum import Enum
import json
import numpy as np
import onnx
from onnx import helper, numpy_helper
from .node_factory import NodeFactory, ensure_opset
from .symbolic_shape_infer import SymbolicShapeInference

class QuantizeConfig:
    def __init__(self, signed, reserved_bits, type_bits):
        self.sign_bit_ = 1 if signed else 0
        self.reserved_bits_ = reserved_bits
        self.type_bits_ = type_bits

    @staticmethod
    def from_dict(qcfg_dict):
        return QuantizeConfig(1 if qcfg_dict['QuantizationType'] == 'Signed' else 0,
                              qcfg_dict['ReservedBit'],
                              qcfg_dict['QuantizeBit'])

    def signed(self):
        return self.sign_bit_ == 1

    def usable_bits(self):
        return self.type_bits_ - self.reserved_bits_

    def q_max(self):
        return float((1 << (self.usable_bits() - self.sign_bit_)) - 1)

    def q_min(self):
        return float(-(self.q_max() + 1) if self.signed() else 0)

    def q_range(self):
        return self.q_max() + 0.5 if self.signed() else float(1 << self.usable_bits())

    def q_type(self):
        if self.type_bits_ == 8:
            return np.int8 if self.sign_bit_ else np.uint8
        else:
            assert self.type_bits_ == 16
            return np.int16 if self.sign_bit_ else np.uint16

    def q_type_bits(self):
        return self.type_bits_

    def __iter__(self): # need this to make dict for json
        return iter([('QuantizeBit', self.type_bits_),
                     ('QuantizationType', 'Signed' if self.sign_bit_ else 'Unsigned'),
                     ('ReservedBit', self.reserved_bits_)])

def quantize_matmul_2d_with_weight(in_node, in_graph, nf, converted_weights, quantized_inputs, qcfg_dict, update_qcfg_dict, default_qcfg, onnx_opset_ver):
    assert in_node.op_type == 'MatMul'

    # quantize weight
    # only handles weight being inputs[1] of MatMul/Gemm node
    fparam_name = in_node.input[1]

    # skip if weights shared by other nodes that's not MatMul
    # TODO: support GEMM op if needed
    other_nodes = [n for n in in_graph.node if n != in_node and fparam_name in n.input and n.op_type != 'MatMul']
    if other_nodes:
        return False

    if in_node.output[0] in qcfg_dict:
        node_qcfg = qcfg_dict[in_node.output[0]]
    else:
        node_qcfg = None
        if not node_qcfg:
            if not update_qcfg_dict and qcfg_dict:
                # when qcfg_dict is readonly, raise warning if qcfg is not found for this node
                print("Warning: qcfg is not found for node with output: " + in_node.output[0] + ", fall back to default qcfg.")
            node_qcfg = default_qcfg

    w_qcfg = QuantizeConfig.from_dict(node_qcfg['W'])
    x_qcfg = QuantizeConfig.from_dict(node_qcfg['X'])
    symmetric = node_qcfg['Symmetric']

    # for symmetric quantization, both weight and input should be quantized to signed
    assert not symmetric or (w_qcfg.signed() and x_qcfg.signed())
    # quantize_type should match between weight and input
    assert w_qcfg.q_type_bits() == x_qcfg.q_type_bits()

    if fparam_name in converted_weights:
        step, base, qparam_rowsum, qparam, w_qcfg1, symmetric1 = converted_weights[fparam_name]
        # for shared weights, node should use the same kind of quantization
        assert dict(w_qcfg1) == dict(w_qcfg)
        assert symmetric1 == symmetric
    else:
        fparam = nf.get_initializer(fparam_name)
        if fparam is None or len(fparam.shape) != 2:
            return False

        q_range = w_qcfg.q_range()
        if symmetric:
            fscale = np.amax(np.abs(fparam), axis=0)
            step = fscale / q_range
            base = 0
        else:
            fmin = np.amin(fparam, axis=0)
            fmax = np.amax(fparam, axis=0)
            fscale = (fmax - fmin)/(2 if w_qcfg.signed() else 1) # signed would be normalized to [-1, 1], and unsigned to [0, 1]
            step = fscale / q_range
            base = (fmax + fmin + step) * 0.5 if w_qcfg.signed() else fmin

        fparam_norm = np.zeros_like(fparam)
        expand_fscale = np.expand_dims(fscale,0)
        np.divide((fparam - np.expand_dims(base,0)), expand_fscale, out=fparam_norm, where=expand_fscale!=0)
        qparam = np.round(fparam_norm * q_range)
        qparam = np.clip(qparam, w_qcfg.q_min(), w_qcfg.q_max())
        qparam_rowsum = np.sum(qparam, axis=0)
        qparam = qparam.astype(w_qcfg.q_type())

        # create new weights in main graph in case other Scans share via converted_weights
        nf.make_initializer(step, fparam_name + '_step', in_main_graph=True)
        nf.make_initializer(qparam, fparam_name + '_qparam', in_main_graph=True)
        step = fparam_name + '_step'
        qparam = fparam_name + '_qparam'
        if symmetric:
            # no need to compute qparam_rowsum and base for symmetric quantization
            base = None
            qparam_rowsum = None
        else:
            nf.make_initializer(base, fparam_name + '_base', in_main_graph=True)
            base = fparam_name + '_base'
            nf.make_initializer(qparam_rowsum, fparam_name + '_qparam_rowsum', in_main_graph=True)
            qparam_rowsum = fparam_name + '_qparam_rowsum'
        converted_weights[fparam_name] = (step, base, qparam_rowsum, qparam, w_qcfg, symmetric)
        nf.remove_initializer(fparam_name)

    # quantize input
    with nf.scoped_prefix(in_node.name) as scoped_prefix:
        input_dim = nf.get_initializer(qparam).shape[0]
        X = in_node.input[0]
        if quantized_inputs is not None:
            quantized_inputs_key = '{}_{}_{}'.format(X, symmetric, '|'.join(['{}:{}'.format(k,v) for (k, v) in x_qcfg]))
        if quantized_inputs is not None and quantized_inputs_key in quantized_inputs:
            scale_X, bias_X, Q_X, Q_X_sum_int32 = quantized_inputs[quantized_inputs_key]
        else:
            if symmetric:
                delta_X = nf.make_node('ReduceMax', nf.make_node('Abs', X), {'axes':[-1]}) # keepdims = 1
                inv_delta_X = nf.make_node('Reciprocal', delta_X)
                norm_X = nf.make_node('Mul', [X, inv_delta_X])
                bias_X = None
                assert x_qcfg.signed()
            else:
                reduce_max_X = nf.make_node('ReduceMax', X, {'axes':[-1]}) # keepdims = 1
                bias_X = nf.make_node('ReduceMin', X, {'axes':[-1]})
                delta_X = nf.make_node('Sub', [reduce_max_X, bias_X])
                inv_delta_X = nf.make_node('Reciprocal', delta_X)
                norm_X = nf.make_node('Mul', [nf.make_node('Sub', [X, bias_X]), inv_delta_X])

            scale_X = nf.make_node('Mul', [delta_X, np.asarray(1.0 / x_qcfg.q_range()).astype(np.float32)])
            Q_Xf = nf.make_node('Mul', [norm_X, np.asarray(x_qcfg.q_range()).astype(np.float32)])
            Q_Xf = nf.make_node('Add', [Q_Xf, np.asarray(0.5).astype(np.float32)])
            Q_Xf = nf.make_node('Floor', Q_Xf)
            if onnx_opset_ver < 11:
                Q_Xf = nf.make_node('Clip', Q_Xf, {'max':x_qcfg.q_max(), 'min':x_qcfg.q_min()})
            else:
                # Clip changed min max to inputs in opset 11
                Q_Xf = nf.make_node('Clip', [Q_Xf, np.asarray(x_qcfg.q_min()).astype(np.float32), np.asarray(x_qcfg.q_max()).astype(np.float32)])
            Q_X = nf.make_node('Cast', Q_Xf, {'to':int({np.uint8  : onnx.TensorProto.UINT8,
                                                        np.int8   : onnx.TensorProto.INT8,
                                                        np.uint16 : onnx.TensorProto.UINT16,
                                                        np.int16  : onnx.TensorProto.INT16}[x_qcfg.q_type()])})

            if symmetric:
                Q_X_sum_int32 = None
            else:
                Q_X_sum_int32 = nf.make_node('ReduceSum', nf.make_node('Cast', Q_X, {'to':int(onnx.TensorProto.INT32)}), {'axes':[-1]})

            if quantized_inputs is not None:
                quantized_inputs[quantized_inputs_key] = (scale_X, bias_X, Q_X, Q_X_sum_int32)

        # MatMulInteger
        if x_qcfg.q_type_bits() == 8:
            Q_Y = nf.make_node('MatMulInteger', [Q_X, qparam])
        else:
            Q_Y = nf.make_node('MatMulInteger16', [Q_X, qparam])
            Q_Y.domain = "com.microsoft"

        # Dequantize
        Y = in_node.output[0]
        if symmetric:
            nf.make_node('Mul',
                      [nf.make_node('Mul', [step, scale_X]),
                       nf.make_node('Cast', Q_Y, {'to': int(onnx.TensorProto.FLOAT)})],
                      output_names=Y)
        else:
            o0 = nf.make_node('Mul', [nf.make_node('Mul', [step, scale_X]),
                                      nf.make_node('Cast', Q_Y, {'to': int(onnx.TensorProto.FLOAT)})])
            o1 = nf.make_node('Mul', [nf.make_node('Mul', [step, bias_X]), qparam_rowsum])
            o2 = nf.make_node('Mul', [base, nf.make_node('Mul', [scale_X, nf.make_node('Cast', Q_X_sum_int32, {'to':int(onnx.TensorProto.FLOAT)})])])
            o3 = nf.make_node('Mul', [base, nf.make_node('Mul', [bias_X, np.asarray(float(input_dim)).astype(np.float32)])])
            nf.make_node('Sum', [o3, o2, o1, o0], output_names=Y)

    if update_qcfg_dict:
        qcfg_dict[in_node.output[0]] = node_qcfg

    return True

def upgrade_op(nf, in_n):
    if in_n.op_type == 'Slice' and len(in_n.input) == 1:
        # convert opset9 Slice to opset10
        with nf.scoped_prefix(in_n.name) as scoped_prefix:
            slice_inputs = [in_n.input[0],
                            np.asarray(NodeFactory.get_attribute(in_n,'starts')).astype(np.int64),
                            np.asarray(NodeFactory.get_attribute(in_n,'ends')).astype(np.int64),
                            np.asarray(NodeFactory.get_attribute(in_n,'axes')).astype(np.int64)]
            nf.make_node('Slice', slice_inputs, output_names=list(in_n.output))
        return True
    elif in_n.op_type == 'TopK' and len(in_n.input) == 1:
        # convert opset1 TopK to opset10
        with nf.scoped_prefix(in_n.name) as scoped_prefix:
            topk_inputs = [in_n.input[0],
                            np.asarray([NodeFactory.get_attribute(in_n,'k')]).astype(np.int64)]
            nf.make_node('TopK', topk_inputs, {'axis':NodeFactory.get_attribute(in_n,'axis',-1)}, output_names=list(in_n.output))
        return True
    else:
        return False

# quantize matmul to MatMulInteger using asymm uint8
def convert_matmul_model(input_model, output_model, only_for_scan=False, share_input_quantization=False, preset_str='asymm8_param0_input1', qcfg_json=None, export_qcfg_json=None):
    preset_qcfgs = {'asymm8_param0_input1' : {'W' : dict(QuantizeConfig(signed=1, reserved_bits=0, type_bits=8)),
                                              'X' : dict(QuantizeConfig(signed=0, reserved_bits=1, type_bits=8)),
                                              'Symmetric' : 0},
                    'symm16_param3_input3' : {'W' : dict(QuantizeConfig(signed=1, reserved_bits=3, type_bits=16)),
                                              'X' : dict(QuantizeConfig(signed=1, reserved_bits=3, type_bits=16)),
                                              'Symmetric' : 1}}
    default_qcfg = preset_qcfgs[preset_str]
    in_mp = onnx.load(input_model)

    qcfg_dict = {}
    if qcfg_json and not export_qcfg_json:
        with open(qcfg_json, 'r') as f:
            qcfg_dict = json.load(f)

    out_mp = onnx.ModelProto()
    out_mp.CopyFrom(in_mp)
    out_mp.ir_version = 5 # update ir version to avoid requirement of initializer in graph input
    onnx_opset_ver = ensure_opset(out_mp, 10) # bump up to ONNX opset 10, which is required for MatMulInteger
    ensure_opset(out_mp, 1, 'com.microsoft') # add MS domain for MatMulInteger16
    out_mp.graph.ClearField('node')
    nf = NodeFactory(out_mp.graph)
    converted_weights = {} # remember MatMul weights that have been converted, in case of sharing
    quantized_inputs = {} if share_input_quantization else None # remember quantized inputs that might be able to share between MatMuls
    for in_n in in_mp.graph.node:
        if upgrade_op(nf, in_n):
            continue

        if in_n.op_type == 'MatMul' and not only_for_scan:
            if quantize_matmul_2d_with_weight(in_n, in_mp.graph, nf, converted_weights, quantized_inputs, qcfg_dict, export_qcfg_json, default_qcfg, onnx_opset_ver):
                continue

        out_n = out_mp.graph.node.add()
        out_n.CopyFrom(in_n)
        if in_n.op_type == 'Scan' or in_n.op_type == 'Loop':
            in_subgraph = NodeFactory.get_attribute(in_n, 'body')
            out_subgraph = NodeFactory.get_attribute(out_n, 'body')
            out_subgraph.ClearField('node')
            scan_nf = NodeFactory(out_mp.graph, out_subgraph)
            subgraph_quantized_inputs = {} if share_input_quantization else None # remember quantized inputs that might be able to share between MatMuls
            for in_sn in in_subgraph.node:
                if in_sn.op_type == 'MatMul':
                    if quantize_matmul_2d_with_weight(in_sn, in_subgraph, scan_nf, converted_weights, subgraph_quantized_inputs, qcfg_dict, export_qcfg_json, default_qcfg, onnx_opset_ver):
                        continue

                if upgrade_op(scan_nf, in_sn):
                    continue

                out_sn = out_subgraph.node.add()
                out_sn.CopyFrom(in_sn)

    onnx.save(out_mp, output_model)
    if export_qcfg_json:
        with open(qcfg_json, 'w') as f:
            f.write(json.dumps(qcfg_dict, indent=2))

def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', required=True, help='The input model file')
  parser.add_argument('--output', required=True, help='The output model file')
  parser.add_argument('--default_qcfg', help='The preset of quantization of <asymm|symm><qbits>_param<reserve_bit>_input<reserve_bit>', choices=['asymm8_param0_input1', 'symm16_param3_input3'], default='asymm8_param0_input1')
  parser.add_argument('--qcfg_json', help='The quantization config json file for read or write.', default=None)
  parser.add_argument('--export_qcfg_json', help='If set, write default quantization config to qcfg_json file.', action='store_true', default=False)
  parser.add_argument('--only_for_scan', help='If set, apply quantization of MatMul only inside scan', action='store_true', default=False)
  parser.add_argument('--share_input_quantization', help='If set, allow input quantization to be shared if the same input is used in multiple MatMul', action='store_true', default=False)
  return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    print('input model: ' + args.input)
    print('output model ' + args.output)
    print('Quantize MatMul to MatMulInteger...')
    assert not args.export_qcfg_json or args.qcfg_json, "--qcfg_json must be specified when --export_qcfg_json is used"
    convert_matmul_model(args.input, args.output, args.only_for_scan, args.share_input_quantization, args.default_qcfg, args.qcfg_json, args.export_qcfg_json)
    print('Done!')

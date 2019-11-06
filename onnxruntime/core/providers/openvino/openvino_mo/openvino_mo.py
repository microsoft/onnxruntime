#!/usr/bin/env python3

"""
  Copyright(C) 2019 Intel Corporation
  Licensed under the MIT License
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from mo.pipeline.common import determined_sort, get_fw_tensor_debug_info, get_sorted_outputs, collect_sub_graphs, relabel_nodes_inplace_safe
from openvino_emitter import port_renumber, serialize_mean_image, create_const_nodes, serialize_network, add_meta_data, generate_ie_ir, serialize_constants, serialize_constants_recursively
import openvino_emitter
from operator import itemgetter
from mo.utils import class_registration
from mo.middle.passes.shape import reverse_input_channels, \
    merge_nodes_permutations, permute_data_nodes_attrs, permute_op_nodes_attrs
from mo.middle.passes.mean_scale_values import move_scaleshift_to_preprocess
from mo.middle.passes.fusing.mark_unfused_nodes import mark_unfused_nodes
from mo.middle.passes.fusing.fuse_linear_seq import fuse_mul_add_sequence
from mo.middle.passes.fusing.fuse_linear_ops import fuse_linear_ops
from mo.middle.passes.fusing.fuse_grouped_conv import grouped_convolutions_fusing
from mo.middle.passes.fusing.decomposition import convert_batch_norm, convert_scale_shift_to_mul_add
from mo.middle.passes.infer import partial_infer, update_fully_connected_shapes
from mo.middle.passes import infer, tensor_names, convert_data_type
from mo.front.onnx.loader import load_onnx_model, protobuf2nx
from mo.front.onnx.extractor import common_onnx_fields, onnx_op_extractor, onnx_op_extractors
from mo.front.extractor import add_output_ops, add_input_ops, \
    extract_node_attrs, create_tensor_nodes, remove_output_ops, user_data_repack
from mo.front.common.register_custom_ops import check_for_duplicates, update_extractors_with_extensions
from extensions.middle.NormalizeFullyConnected import NormalizeFullyConnected
from extensions.middle.EltwiseInputNormalization import EltwiseInputNormalize
from mo.utils.versions_checker import check_requirements
from mo.utils.version import get_version
from mo.utils.utils import refer_to_faq_msg
from mo.utils.logger import init_logger
from mo.utils.guess_framework import guess_framework_by_ext
from mo.utils.error import Error, FrameworkError
from mo.utils.cli_parser import get_placeholder_shapes, get_tuple_values, get_model_name, \
    get_common_cli_options, get_caffe_cli_options, get_tf_cli_options, get_mxnet_cli_options, get_kaldi_cli_options, \
    get_onnx_cli_options, get_mean_scale_dictionary, parse_tuple_pairs, get_meta_info
from mo.utils.versions_checker import check_python_version
import onnx
import numpy as np
import networkx as nx
from collections import OrderedDict
import traceback
import logging as log
import datetime
import argparse
import sys
import os

ov_root = os.environ['INTEL_CVSDK_DIR']
if '2019.3' in ov_root:
    version = '2019.R3'
else:
    version = 'unsupported'
    print('You are using unsupported version of OpenVINO. Please refer to BUILD.md for supported versions of OpenVINO.')
mo_path = ov_root + "/deployment_tools/model_optimizer"
print('mo_path = {}'.format(mo_path))
mo_extensions = mo_path + "/extensions"
sys.path.append(mo_path)

from extensions.middle.quantize_fuses import MarkNodesToFuseUpToFakeQuantize, FakeQuantizeFuse
from extensions.back.CreateConstNodes import CreateConstNodesReplacement
from extensions.back.RemoveRedundantReshapes import RemoveRedundantReshapes
from extensions.back.FuseReshapesSequence import FuseReshapesSequence
from extensions.middle.AddFakeQuantizeFuse import AddFakeQuantizeFuse
from extensions.middle.MulFakeQuantizeFuse import MulFakeQuantizeFuse
from mo.middle.pattern_match import for_graph_and_each_sub_graph_recursively
from mo.graph.graph import check_empty_graph, Node, Graph
from mo.utils import import_extensions, class_registration
from mo.middle.passes.conv import fuse_pad
from mo.middle.passes.eliminate import remove_const_ops, mark_output_reachable_nodes, mark_undead_nodes, mark_const_producer_nodes, \
    eliminate_dead_nodes, add_constant_operations, shape_inference, remove_op_nodes, get_nodes_with_attributes
from mo.middle.passes.conv import convert_muladd_to_scaleshift, \
    convert_add_or_mul_to_scaleshift, \
    convert_matmul_to_fully_connected, batch_norm_fuse
from mo.middle.passes.fusing.decomposition import convert_scale_shift_to_mul_add, convert_batch_norm


def is_fully_defined_shape(shape: np.ndarray):
    if -1 in shape:
        return True
    return True


infer.is_fully_defined_shape = is_fully_defined_shape


def prepare_emit_ir(graph: nx.MultiDiGraph, data_type: str, output_dir: str, output_model_name: str,
                    mean_data: [list, None] = None, input_names: list = [], meta_info: dict = dict()):

  
    for sub_graph in [graph] + collect_sub_graphs(graph):
        create_const_nodes(
            sub_graph, start_data_nodes_are_not_allowed=(sub_graph == graph))
 
        op_order, data_order = determined_sort(get_sorted_outputs(sub_graph))
        mapping = {v: u for u, v in enumerate(op_order)}
        mapping.update({v: u for u, v in enumerate(
            data_order, start=len(sub_graph))})
        relabel_nodes_inplace_safe(sub_graph, mapping)
        port_renumber(sub_graph)
        convert_data_type.convert(sub_graph, data_type)

    tensor_names.propagate_op_name_to_tensor(graph)
    weights = np.array([])
    bin_file = os.path.join(output_dir, '{}.bin'.format(output_model_name))
    if(data_type == "FP16"):
        weights = serialize_constants(weights, graph, data_type=np.float16)
    elif(data_type == "FP32"):
        weights = serialize_constants(weights, graph, data_type=np.float32)


    mean_offset = None
    mean_size = None
    if mean_data:
        mean_offset, mean_size = serialize_mean_image(
            bin_file, mean_data=mean_data)
   
    xml_string = generate_ie_ir(graph=graph,
                                file_name=os.path.join(
                                    output_dir, '{}.xml'.format(output_model_name)),
                                input_names=input_names,
                                mean_offset=mean_offset,
                                mean_size=mean_size,
                                meta_info=meta_info)

    return weights, xml_string


def graph_clean_up(graph: Graph, undead_node_types: list = None):
 
    if undead_node_types is None:
        undead_node_types = []

    if 'Shape' in undead_node_types:
        undead_node_types.remove('Shape')

    mark_output_reachable_nodes(graph)
    mark_undead_nodes(graph, undead_node_types)
    mark_const_producer_nodes(graph)
    eliminate_dead_nodes(graph)
    # Add Const op for constant data nodes
    add_constant_operations(graph)
    shape_inference(graph)
       

def graph_clean_up_onnx(graph: Graph):
    graph_clean_up(graph, ['Shape'])


def driver(onnx_modelproto_bytes, precision: str, output_model_name: str, output_dir: str): 
    try:
        model_proto = onnx.load_from_string(bytes(onnx_modelproto_bytes))
    except Exception as e:
        print("[python] onnx exception: ", str(e))

    model_graph = model_proto.graph  # pylint: disable=no-member
    log.debug("Number of nodes in graph_def: {}".format(len(model_graph.node)))
    log.debug("Number of all input ports (not true inputs) in graph_def: {}".format(len(model_graph.input)))
    log.debug("Number of initializers in graph_def: {}".format(len(model_graph.initializer)))
    log.debug("Number of real inputs in graph_def: {}".format(len(model_graph.input) - len(model_graph.initializer)))
    update_extractors_with_extensions(onnx_op_extractors)

    try:
        graph = protobuf2nx(model_proto)
        log.debug("Number of nodes in NX graph: {}".format(graph.number_of_nodes()))
        graph.__setattr__('name',
                          output_model_name if output_model_name else model_proto.graph.name)  # pylint: disable=no-member
        graph.graph['layout'] = 'NCHW'
        graph.graph['fw'] = 'onnx'
        graph.graph['feature_dim'] = 1 if graph.graph['layout'] == 'NCHW' else 3
        graph.graph['cmd_params'] = argparse.Namespace(batch=None, data_type='float', disable_fusing=False, disable_gfusing=False, disable_resnet_optimization=False, enable_concat_optimization=False, extensions=mo_extensions, finegrain_fusing=None, framework='onnx', freeze_placeholder_with_value=None, generate_deprecated_IR_V2=False,
                                            input=None, input_model=None, input_shape=None, keep_shape_ops=False, log_level='ERROR', mean_scale_values={}, mean_values=(), model_name=None, move_to_preprocess=False, output=None, output_dir='.', placeholder_shapes=None, reverse_input_channels=False, scale=None, scale_values=(), silent=False, version=False,
                                            blobs_as_inputs=False,keep_quantize_ops_in_IR=False,generate_experimental_IR_V10=False)
        graph.graph['ir_version'] = 6

    except Exception as e:
        raise Error(
            'Cannot pre-process ONNX graph after reading from model file "{}". ' \
            'File is corrupt or has unsupported format. Details: {}. ' +
            refer_to_faq_msg(44),
            model_file_name,
            str(e)
        ) from e
    graph.check_empty_graph('protobuf2nx. It may happen due to problems with loaded model')
    extract_node_attrs(graph, lambda node: onnx_op_extractor(node, check_for_duplicates(onnx_op_extractors)))

    # --------------------------------- LOAD END ------------------------------------------------------
    class_registration.apply_replacements(graph, class_registration.ClassType.FRONT_REPLACER)
    class_registration.apply_replacements(graph, class_registration.ClassType.MIDDLE_REPLACER)

    fuse_pad(graph)
    graph_clean_up_onnx(graph)

    for_graph_and_each_sub_graph_recursively(graph, convert_matmul_to_fully_connected)

    # Mark nodes with attr 'can_be_fused': False to disable fusing for specified nodes
    mark_unfused_nodes(graph, False)

    # Converting FusedBatchNorm layer to Mul->Add->Mul->Add sequence
    # IE doesn't support BN with 4 inputs, so we have to split it to two ScaleShift
    convert_batch_norm(graph)
    graph_clean_up_onnx(graph)

    # Converting ScaleShift layer to Mul->Add
    convert_scale_shift_to_mul_add(graph)
    graph_clean_up_onnx(graph)

    # Fusing the sequences of Mul/Add operations
    fuse_mul_add_sequence(graph)
    graph_clean_up_onnx(graph)

    # Fusing linear operation to Convolution
    fuse_linear_ops(graph)
    graph_clean_up_onnx(graph)

    grouped_convolutions_fusing(graph)
    graph_clean_up_onnx(graph)

    fuse_linear_ops(graph)
    graph_clean_up_onnx(graph)

    MarkNodesToFuseUpToFakeQuantize().find_and_replace_pattern(graph)
    FakeQuantizeFuse().find_and_replace_pattern(graph)

    AddFakeQuantizeFuse().find_and_replace_pattern(graph)
    MulFakeQuantizeFuse().find_and_replace_pattern(graph)

    convert_muladd_to_scaleshift(graph)
    graph_clean_up_onnx(graph)

    graph_clean_up_onnx(graph)
    convert_add_or_mul_to_scaleshift(graph)  # scale = 1
    graph_clean_up_onnx(graph)

    fuse_pad(graph)
    graph_clean_up_onnx(graph)

    FuseReshapesSequence().find_and_replace_pattern(graph)
    RemoveRedundantReshapes().find_and_replace_pattern(graph)

    graph_clean_up_onnx(graph)

    pattern = EltwiseInputNormalize()
    pattern.find_and_replace_pattern(graph)

    merge_nodes_permutations(graph)
    permute_data_nodes_attrs(graph)
    permute_op_nodes_attrs(graph)

    graph_clean_up_onnx(graph)
    class_registration.apply_replacements(graph, class_registration.ClassType.BACK_REPLACER)

    for_graph_and_each_sub_graph_recursively(graph, remove_const_ops)

    CreateConstNodesReplacement().find_and_replace_pattern(graph)

    for_graph_and_each_sub_graph_recursively(graph, remove_output_ops)
    
    weights, xml_string = prepare_emit_ir(graph=graph, data_type=precision, output_dir=output_dir, output_model_name=output_model_name,
                    meta_info={'unset': []})

    
    return weights, xml_string



def driver_entry(onnx_modelproto_bytes, precision: str):
    start_time = datetime.datetime.now()

    model_name = "optimized_model"
    outputs = None
    placeholder_shapes = {}
    mean_values = {}
    scale_values = {}
    mean_scale = {}

    from mo.front.onnx.register_custom_ops import get_front_classes
    import_extensions.load_dirs('onnx', [mo_extensions], get_front_classes)
    weights, xml_string = driver(onnx_modelproto_bytes, precision, model_name, ".")

    return weights, xml_string


def convert_fp16(onnx_modelproto_bytes):
    try:
        init_logger('ERROR', False)
        framework = 'onnx'

        weights, xml_string = driver_entry(
            onnx_modelproto_bytes, precision='FP16')

        float_array = np.asarray(weights, dtype=np.float16)

        return float_array, xml_string
    except:
        return 1


def convert_fp32(onnx_modelproto_bytes):
    try:
        init_logger('ERROR', False)
        framework = 'onnx'

        weights, xml_string = driver_entry(
            onnx_modelproto_bytes, precision='FP32')

        float_array = np.asarray(weights, dtype=np.float32)

        return float_array, xml_string
    except:
        return 1


if __name__ == "__main__":
    ret_code = check_python_version()
    if ret_code:
        sys.exit(ret_code)

    from mo.utils.cli_parser import get_onnx_cli_parser
    
    weights_string, final_string = convert_fp32()
 
    sys.exit(0)

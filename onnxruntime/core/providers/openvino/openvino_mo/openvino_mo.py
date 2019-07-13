#!/usr/bin/env python3

"""
  Copyright(C) 2019 Intel Corporation
  Licensed under the MIT License
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import os
ov_root = os.environ['INTEL_OPENVINO_DIR']
mo_path = ov_root + "/deployment_tools/model_optimizer"
mo_extensions = mo_path + "/extensions"
sys.path.append(mo_path)

import argparse
import datetime
import logging as log
import traceback
import numpy as np
import logging as log
import networkx as nx
import onnx
from collections import OrderedDict
from operator import itemgetter

from mo.utils.versions_checker import check_python_version
from mo.utils import import_extensions, class_registration
from mo.utils.error import Error, FrameworkError
from mo.utils.guess_framework import guess_framework_by_ext
from mo.utils.logger import init_logger
from mo.utils.utils import refer_to_faq_msg
from mo.utils.version import get_version
from mo.utils.versions_checker import check_requirements
from extensions.middle.EltwiseInputNormalization import EltwiseInputNormalize
from extensions.middle.NormalizeFullyConnected import NormalizeFullyConnected
from mo.front.common.register_custom_ops import check_for_duplicates, update_extractors_with_extensions
from mo.front.extractor import add_output_ops, add_input_ops, \
    extract_node_attrs, create_tensor_nodes, remove_output_ops, user_data_repack
from mo.front.onnx.extractor import common_onnx_fields, onnx_op_extractor, onnx_op_extractors
from mo.front.onnx.loader import load_onnx_model, protobuf2nx
from mo.middle.passes import tensor_names, convert_data_type
from mo.middle.passes.conv import convert_add_or_mul_to_scaleshift, convert_muladd_to_scaleshift_or_power, fuse_pad
from mo.middle.passes.fusing.decomposition import convert_batch_norm, convert_scale_shift_to_mul_add
from mo.middle.passes.fusing.fuse_grouped_conv import grouped_convolutions_fusing
from mo.middle.passes.fusing.fuse_linear_ops import fuse_linear_ops
from mo.middle.passes.fusing.fuse_linear_seq import fuse_mul_add_sequence
from mo.middle.passes.fusing.mark_unfused_nodes import mark_unfused_nodes
from mo.middle.passes.infer import override_placeholder_shapes, convert_mul_add_to_power,  override_batch, exit_bound_edges, control_flow_infer
from mo.middle.pattern_match import for_graph_and_each_sub_graph_recursively
from extensions.back.CreateConstNodes import CreateConstNodesReplacement
from mo.middle.passes.eliminate import remove_const_ops, mark_output_reachable_nodes, mark_undead_nodes, mark_const_producer_nodes, \
    eliminate_dead_nodes, add_constant_operations, shape_inference, remove_op_nodes, get_nodes_with_attributes
from mo.middle.passes.mean_scale_values import move_scaleshift_to_preprocess
from mo.middle.passes.shape import convert_reshape, reverse_input_channels, \
    fuse_sequence_of_reshapes, merge_nodes_permutations, permute_data_nodes_attrs, permute_op_nodes_attrs
from mo.graph.graph import check_empty_graph, Node, Graph
from openvino_emitter import port_renumber, serialize_mean_image, create_const_nodes, serialize_network, add_meta_data, generate_ie_ir, serialize_constants, serialize_constants_recursively
from mo.pipeline.common import determined_sort, get_fw_tensor_debug_info, get_sorted_outputs, collect_sub_graphs, relabel_nodes_inplace_safe
from mo.middle.passes import infer

def is_fully_defined_shape(shape: np.ndarray):
    if -1 in shape:
        return True
    return True

infer.is_fully_defined_shape = is_fully_defined_shape

def partial_infer(graph: nx.MultiDiGraph, start_node: str = None):

    print("In partial Infer")
    cycle_nodes = get_nodes_with_attributes(graph, is_cyclic=True)
    cycle_nodes = [Node(graph, node).out_node().id for node in cycle_nodes]
    ebunch_cyclic = list(graph.out_edges(nbunch=cycle_nodes, data=True, keys=True))
    ebunch_reconnected = exit_bound_edges(graph, sources=cycle_nodes, end_node_attrs={'op': 'Exit'})
    graph.remove_edges_from(ebunch_cyclic)
    graph.add_edges_from(ebunch_reconnected)

    try:
        nodes = list(nx.topological_sort(graph))
    except:
        raise Error('Graph contains a cycle. Can not proceed. ' + refer_to_faq_msg(97))

    graph.remove_edges_from(ebunch_reconnected)
    graph.add_edges_from(ebunch_cyclic)

    # Mark all nodes as not inferred yet
    if not start_node is None:
        start_index = nodes.index(start_node)
        nx.set_node_attributes(G=graph.subgraph(nodes[start_index:]), name='is_partial_inferred', values=False)
    else:
        nx.set_node_attributes(G=graph, name='is_partial_inferred', values=False)

    nx.set_node_attributes(G=graph, name='executable',
                           values={n: True for n in get_nodes_with_attributes(graph, kind='data')})

    for n in nodes:
        # Data Flow Infer
        try:
            node = Node(graph, n)
            node_name = node.soft_get('name')
            if node.has('is_partial_inferred') and not node.is_partial_inferred:
                if node.has('infer') and not node.infer is None:
                    node.infer(node)
                    out_nodes = node.out_nodes()

                    # propagate nchw_layout attributes to data nodes
                    if node.has('nchw_layout'):
                        for out_node in out_nodes.values():
                            out_node['nchw_layout'] = node.nchw_layout

                    for out_port, out_node in out_nodes.items():
                        not_all_output_shapes = False
                        if not out_node.has_valid('shape'):
                            not_all_output_shapes = True
                        elif not is_fully_defined_shape(out_node.shape):
                            not_all_output_shapes = True

                node.is_partial_inferred = True

        except Exception as err:
            raise Error('Stopped shape/value propagation at "{}" node. '.format(node.soft_get('name')) +
                        refer_to_faq_msg(38)) from err
        control_flow_infer(graph, n)

    not_fully_inferred = get_nodes_with_attributes(graph, is_not_fully_inferred=True)
    for n in not_fully_inferred:
        node = Node(graph, n)
        if node.has('infer') and not node.infer is None:
            node.infer(node)

    #delete_not_executable(graph)
    return graph

def update_fully_connected_shapes(graph: nx.MultiDiGraph):
    nodes = nx.topological_sort(graph)
    while True:
        should_infer = False
        for n in nodes:
            node = Node(graph, n)
            if node.has('type') and node.type == 'FullyConnected' and node.in_node(0).shape.size == 3:
                log.debug("node.in_node(0).shape = {}".format(node.in_node(0).shape))
                log.debug("channel_dims = {}".format(node.channel_dims))
                assert (node.in_node(0).shape.size == 3 and node.channel_dims > 0)
                node.in_node(0).shape = np.delete(node.in_node(0).shape, 1)
                if node.out_node().shape.size == 3:
                    node.channel_dims = node.channel_dims - 1
                    log.debug("Initiated partial infer from update_fully_connected_shapes")
                    graph = partial_infer(graph, node.in_node(0).id)
                    should_infer = True
                    break
        if not should_infer:
            break

def prepare_emit_ir(graph: nx.MultiDiGraph, data_type: str, output_dir: str, output_model_name: str,
                    mean_data: [list, None] = None, input_names: list = [], meta_info: dict = dict()):

    for sub_graph in [graph] + collect_sub_graphs(graph):
        #create_const_nodes(sub_graph, start_data_nodes_are_not_allowed=(sub_graph == graph))
        op_order, data_order = determined_sort(get_sorted_outputs(sub_graph))
        mapping = {v: u for u, v in enumerate(op_order)}
        mapping.update({v: u for u, v in enumerate(data_order, start=len(sub_graph))})
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
    #weights = serialize_constants(weights, graph)

    mean_offset = None
    mean_size = None
    if mean_data:
        mean_offset, mean_size = serialize_mean_image(bin_file, mean_data=mean_data)

    xml_string = generate_ie_ir(graph=graph,
                   file_name=os.path.join(output_dir, '{}.xml'.format(output_model_name)),
                   input_names=input_names,
                   mean_offset=mean_offset,
                   mean_size=mean_size,
                   meta_info=meta_info)

    return weights, xml_string

def graph_clean_up(graph: Graph, undead_node_types: list = None):
    if undead_node_types is None:
        undead_node_types = []

    if 'Shape' in undead_node_types:#and not graph.graph['cmd_params'].keep_shape_ops:
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

def driver(onnx_modelproto_bytes, precision : str, output_model_name: str, outputs: list, output_dir: str,
           scale: float,
           user_shapes: [None, list, np.array] = None,
           mean_scale_values: [dict, list] = ()):

    try:
        model_proto = onnx.load_from_string(bytes(onnx_modelproto_bytes))
    except Exception as e:
        print("[python] onnx exception: ", str(e))

    model_graph = model_proto.graph  # pylint: disable=no-member

    update_extractors_with_extensions(onnx_op_extractors)

    try:
        graph = protobuf2nx(model_proto)
        log.debug("Number of nodes in NX graph: {}".format(graph.number_of_nodes()))
        graph.__setattr__('name', output_model_name if output_model_name else model_proto.graph.name)  # pylint: disable=no-member
        graph.graph['layout'] = 'NCHW'
        graph.graph['cmd_params'] = argparse.Namespace(batch=None, data_type='float', disable_fusing=False, disable_gfusing=False, disable_resnet_optimization=False, enable_concat_optimization=False, extensions=mo_extensions, finegrain_fusing=None, framework='onnx', freeze_placeholder_with_value=None, generate_deprecated_IR_V2=False, input=None, input_model=None, input_shape=None, keep_shape_ops=False, log_level='ERROR', mean_scale_values={}, mean_values=(), model_name=None, move_to_preprocess=False, output=None, output_dir='.', placeholder_shapes=None, reverse_input_channels=False, scale=None, scale_values=(), silent=False, version=False)
        graph.graph['fw'] = 'onnx'
        graph.graph['feature_dim'] = 1 if graph.graph['layout'] == 'NCHW' else 3
        graph.graph['ir_version'] = 4
        extract_node_attrs(graph, lambda node: (True, common_onnx_fields(node)))
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


    mark_unfused_nodes(graph, 'False')
    convert_batch_norm(graph)
    graph_clean_up_onnx(graph)


    #AddQuantizeFuse().find_and_replace_pattern(graph)
    #MulQuantizeFuse().find_and_replace_pattern(graph)

    convert_muladd_to_scaleshift_or_power(graph)
    graph_clean_up_onnx(graph)

    convert_mul_add_to_power(graph)
    graph_clean_up_onnx(graph)

    convert_reshape(graph)
    graph_clean_up_onnx(graph)
    convert_add_or_mul_to_scaleshift(graph)  # scale = 1
    graph_clean_up_onnx(graph)

    fuse_pad(graph)
    graph_clean_up_onnx(graph)

    fuse_sequence_of_reshapes(graph)
    graph_clean_up_onnx(graph)

    pattern = EltwiseInputNormalize()
    pattern.find_and_replace_pattern(graph)

    merge_nodes_permutations(graph)
    permute_data_nodes_attrs(graph)
    permute_op_nodes_attrs(graph)

    class_registration.apply_replacements(graph, class_registration.ClassType.BACK_REPLACER)

    for_graph_and_each_sub_graph_recursively(graph, remove_const_ops)

    CreateConstNodesReplacement().find_and_replace_pattern(graph)

    for_graph_and_each_sub_graph_recursively(graph, remove_output_ops)

    weights, xml_string = prepare_emit_ir(graph=graph, data_type=precision, output_dir=output_dir, output_model_name=output_model_name,
                    meta_info={'unset': []})

    return weights, xml_string


def driver_entry(onnx_modelproto_bytes, precision : str):
    model_name = "optimized_model"
    outputs = None
    placeholder_shapes = {}
    mean_values = {}
    scale_values = {}
    mean_scale = {}
    from mo.utils.class_registration import update_registration
    #from mo.front.onnx.register_custom_ops import update_registration


    from mo.front.onnx.register_custom_ops import get_front_classes
    import_extensions.load_dirs('onnx', [mo_extensions], get_front_classes)

    weights , xml_string = driver(onnx_modelproto_bytes, precision, model_name, outputs, ".", None,
                             user_shapes=placeholder_shapes,
                             mean_scale_values=mean_scale)

    return weights, xml_string


def convert_fp16(onnx_modelproto_bytes):
    try:
        init_logger('ERROR', False)
        framework = 'onnx'

        weights, xml_string = driver_entry(onnx_modelproto_bytes, precision='FP16')

        float_array = np.asarray(weights, dtype=np.float16)


        return float_array, xml_string
    except: #(FileNotFoundError, NotADirectoryError) as e:
        #log.error('File {} was not found'.format(str(e).split('No such file or directory:')[1]))
        #log.debug(traceback.format_exc())
        return 1

def convert_fp32(onnx_modelproto_bytes):
    try:
        init_logger('ERROR', False)
        framework = 'onnx'

        weights, xml_string = driver_entry(onnx_modelproto_bytes, precision='FP32')
        #weights_string = np.array2string(weights, precision=10, separator="")

        float_array = np.asarray(weights, dtype=np.float32)

        return float_array, xml_string
    except: #(FileNotFoundError, NotADirectoryError) as e:
        #log.error('File {} was not found'.format(str(e).split('No such file or directory:')[1]))
        #log.debug(traceback.format_exc())
        return 1

if __name__ == "__main__":
    ret_code = check_python_version()
    if ret_code:
        sys.exit(ret_code)

    from mo.utils.cli_parser import get_onnx_cli_parser
    weights_string, final_string = convert_fp32()
    sys.exit(0)





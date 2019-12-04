"""
  Copyright(C) 2019 Intel Corporation
  Licensed under the MIT License
"""

from mo.utils.version import get_version
from mo.utils.utils import refer_to_faq_msg
from mo.utils.unsupported_ops import UnsupportedOps
from mo.graph.graph import *
from mo.front.extractor import update_ie_fields
from xml.etree.ElementTree import Element, SubElement, tostring
import xml.dom.minidom
import hashlib
import sys
import os
ov_root = os.environ['INTEL_CVSDK_DIR']
mo_path = ov_root + "/deployment_tools/model_optimizer"
sys.path.append(mo_path)


def create_const_nodes(graph: nx.MultiDiGraph, start_data_nodes_are_not_allowed: bool = True):

    for node_name in list(graph.nodes()):
        node = Node(graph, node_name)
        if (
                node.has('kind') and
                node.kind == 'data' and (
                (len(node.out_edges()) == 1 and 'bin' not in node.out_edge(0)) or
                node.has_and_set('is_output')
                ) and
                len(node.in_nodes()) == 0):

            if node.has_valid('value'):
                const_node_name = node.id + '_const'
                log.debug("Added Const node '{}'".format(const_node_name))
                graph.add_node(const_node_name, name=const_node_name, type='Const', kind='op', op='Const',
                               precision="FP32")
                update_ie_fields(node.graph.node[const_node_name])
                graph.add_edges_from([(const_node_name, node.id, {'out': 0})])
                copy_data_node_name = unique_id(graph, node.id + '_copy_')
                graph.add_node(copy_data_node_name, kind='data', precision="FP32", shape=np.array(node.shape),
                               value=np.array(node.value))
                if node.has_valid('force_precision'):
                    Node(graph, copy_data_node_name)[
                        'force_precision'] = node.force_precision
                    Node(graph, const_node_name)[
                        'force_precision'] = node.force_precision
                graph.add_edges_from(
                    [(copy_data_node_name, const_node_name, {'in': 0, 'bin': 'custom'})])
            elif start_data_nodes_are_not_allowed:
                log.debug('node = {}'.format(node.graph.node[node.id]))
                # TODO for body sub-graph it shouldn't be reported as an error
                raise Error(
                    'Discovered data node without inputs and value, node.name = {}, consumer.name = {}. ' +
                    refer_to_faq_msg(23),
                    node.soft_get('name'),
                    node.out_node().soft_get('name') if len(node.out_nodes()) else "<no consumer>"
                )


def serialize_constants(weights, graph: nx.MultiDiGraph,  data_type=np.float32):

    bin_hashes = {}

    weights = serialize_constants_recursively(
        weights, graph, data_type, bin_hashes)

    return weights


def serialize_constants_recursively(weights, graph: nx.MultiDiGraph, data_type, bin_hashes):
    nodes = sorted(graph.nodes())
    weights = []
    start = 0
    precision = 0
    if(data_type == np.float32):
        precision = 4
    elif (data_type == np.float16):
        precision = 2
    for node in nodes:
        node = Node(graph, node)

        if node.kind == 'data' and node.value is not None and any('bin' in d for u, v, d in graph.out_edges(node.node, data=True)):
            blob = node.value
            blob_hash = hashlib.sha512(blob.tobytes()).hexdigest()

            if blob_hash in bin_hashes and np.array_equal(blob, bin_hashes[blob_hash]['blob']):
                graph.node[node.node]['offset'] = bin_hashes[blob_hash]['offset']
                graph.node[node.node]['size'] = bin_hashes[blob_hash]['size']
            else:

                end = blob.size * precision + start
                blob_flatten = blob.flatten()
                weights = np.append(weights, blob_flatten)

                graph.node[node.node]['offset'] = start
                graph.node[node.node]['size'] = end - start
                start = start + blob.size * precision

                bin_hashes[blob_hash] = {'offset': graph.node[node.node]['offset'],
                                         'size': graph.node[node.node]['size'], 'blob': blob}

            log.debug(
                "Detected binary for graph: '{}', node: '{}', id: {}, shape: '{}', offset: '{}', size: '{}'".format(
                    graph, node.soft_get('name'), node.id, node.shape, node.offset, node.size))

    for node in nodes:
        node = Node(graph, node)
        # Dump blobs recursively if sub-graphs are present in the node
        if node.has_valid('sub_graphs'):
            for sub_graph_attr_name in node.sub_graphs:
                sub_graph = node[sub_graph_attr_name]
                weights = serialize_constants_recursively(
                    weights, sub_graph, data_type, bin_hashes)

    return weights


def serialize_mean_image(bin_file_name: str, mean_data=[]):
    with open(bin_file_name, 'ab') as bin_file:
        mean_offset = []
        mean_size = []
        for x in range(len(mean_data)):
            start = bin_file.tell()
            bin_file.write(mean_data[x][:])
            end = bin_file.tell()
            mean_offset.append(start)
            mean_size.append(end - start)

        return mean_offset, mean_size


def xml_shape(shape: np.ndarray, element: xml.etree.ElementTree.Element):
    for d in shape:
        dim = SubElement(element, 'dim')
        if d <= 0:
            d = 1
           # raise Error('The value "{}" for shape is less or equal to 0. May be the input shape of the topology is '
           #         'wrong.'.format(d))
        if int(d) != d:
            raise Error('The value "{}" for shape is not integer.'.format(d))
        if not isinstance(d, np.int64):
            log.warning(
                'The element of shape is not np.int64 value. Converting the value "{}" to integer'.format(d))
            d = int(d)
        dim.text = str(d)


def sorted_inputs(node):
    return node.get_sorted_inputs(node)


def sorted_outputs(node):
    return node.get_sorted_outputs(node)


def xml_ports(node: Node, element: xml.etree.ElementTree.Element, edges: xml.etree.ElementTree.Element):
    # input ports
    inputs = None  # will create input section only if at least one input is available
    for u, d in sorted_inputs(node):
        if 'bin' not in d and ('xml_skip' not in d or not d['xml_skip']):
            if inputs is None:
                inputs = SubElement(element, 'input')
            port = SubElement(inputs, 'port')
            port.set('id', str(d['in']))
            assert node.graph.node[u]['shape'] is not None, 'Input shape is not calculated properly for node {}'.format(
                node.id)
            xml_shape(node.graph.node[u]['shape'], port)
            # u is a data node that has a single producer, let's find it
            assert (node.graph.node[u]['kind'] == 'data')
            in_nodes = list(node.graph.in_edges(u, data=True))
            assert (len(in_nodes) <= 1)
            if len(in_nodes) == 1:
                src, _, out_attrs = in_nodes[0]
                edge = SubElement(edges, 'edge')
                edge.set('from-layer', str(src))
                edge.set('from-port', str(out_attrs['out']))
                edge.set('to-layer', str(node.node))
                edge.set('to-port', str(d['in']))

    # output ports
    outputs = None
    for v, d in sorted_outputs(node):
        if 'xml_skip' not in d or not d['xml_skip']:
            if outputs is None:
                outputs = SubElement(element, 'output')
            port = SubElement(outputs, 'port')
            port.set('id', str(d['out']))
            assert node.graph.node[v][
                'shape'] is not None, 'Output shape is not calculated properly for node {}'.format(
                node.id)
            xml_shape(node.graph.node[v]['shape'], port)


def xml_consts(graph: nx.MultiDiGraph, node: Node, element: xml.etree.ElementTree.Element):
    blobs = None  # sub-element that will be created on-demand
    for u, d in sorted_inputs(node):
        if 'bin' in d:
            if not blobs:
                blobs = SubElement(element, 'blobs')
            const = SubElement(blobs, d['bin'])
            try:
                const.set('offset', str(graph.node[u]['offset']))
                const.set('size', str(graph.node[u]['size']))
            except Exception as e:
                raise Error('Unable to access binary attributes ("offset" and/or "size") '
                            'for blobs for node {}. Details: {}'.format(node.soft_get('name'), e))


def soft_get(node, attr):
    ''' If node has soft_get callable member, returns node.soft_get(attr), else return <SUB-ELEMENT> '''
    return node.soft_get(attr) if hasattr(node, 'soft_get') and callable(node.soft_get) else '<SUB-ELEMENT>'


def serialize_element(
        graph: nx.MultiDiGraph,
        node,
        schema: list,
        parent_element: xml.etree.ElementTree.Element,
        edges: xml.etree.ElementTree.Element,
        unsupported):

    name, attrs, subelements = schema
    element = SubElement(parent_element, name)
    for attr in attrs:
        if isinstance(attr, tuple):
            key = attr[0]
            try:
                if callable(attr[1]):
                    value = attr[1](node)
                else:
                    value = node[attr[1]] if attr[1] in node else None
            except TypeError as e:
                raise Error('Unable to extract {} from layer {}',
                            key, soft_get(node, 'name')) from e
            except Exception as e:
                raise Error(
                    'Cannot emit value for attribute {} for layer {}. '
                    'Internal attribute template: {}.',
                    key,
                    soft_get(node, 'name'),
                    attr
                ) from e
        elif isinstance(attr, dict):
            node_attrs = node.graph.node[node.id] if isinstance(
                node, Node) else node
            for key in attr.keys():
                if key in node_attrs:
                    for k, v in node_attrs[key].items():
                        element.set(k, str(v))
            continue
        else:
            key = attr
            value = node[attr] if attr in node else None
        if value is not None:
            element.set(key, str(value))
    serialize_node_attributes(graph, node, subelements,
                              element, edges, unsupported)
    if len(element.attrib) == 0 and len(element.getchildren()) == 0:
        parent_element.remove(element)


def serialize_meta_list(graph, node, schema, element, edges, unsupported):
    _, list_accessor, sub_schema = schema
    items = list_accessor(node)  # this is a list of dictionary-like objects
    for item in items:
        serialize_node_attributes(
            graph, item, [sub_schema], element, edges, unsupported)


def serialize_node_attributes(
        graph: nx.MultiDiGraph,  # the current network graph
        node,   # dictionry-like object that should be serialized
        schema: list,
        parent_element: xml.etree.ElementTree.Element,
        edges: xml.etree.ElementTree.Element,
        unsupported):

    try:
        for s in schema:
            if not isinstance(s, tuple):
                if s == '@ports':
                    try:
                        # TODO make sure that edges are generated regardless of the existence of @ports
                        xml_ports(node, parent_element, edges)
                    except Exception as e:
                        raise Error(('Unable to create ports for node with id {}. ' +
                                     refer_to_faq_msg(3)).format(node.id)) from e
                elif s == '@consts':
                    xml_consts(graph, node, parent_element)
                else:
                    log.warning('Unknown xml schema tag: {}'.format(s))
            else:
                name = s[0]
                if name == '@list':
                    serialize_meta_list(
                        graph, node, s, parent_element, edges, unsupported)
                elif name == '@network':
                    serialize_network(node[s[1]], parent_element, unsupported)
                else:
                    serialize_element(
                        graph, node, s, parent_element, edges, unsupported)
    except Exception as e:
        raise Error(
            'Error while emitting attributes for layer {} (id = {}). '
            'It usually means that there is unsupported pattern around this node or unsupported combination of attributes.',
            soft_get(node, 'name'),
            node.id
        ) from e


def create_pre_process_block_for_image(net: xml.etree.ElementTree.Element, ref_layer_names: list, mean_offset: tuple,
                                       mean_size: tuple):
    pre_process = SubElement(net, 'pre-process')
    # TODO: to think about need to output FP16 mean values
    pre_process.set('mean-precision', 'FP32')
    # TODO: extend it for several inputs
    pre_process.set('reference-layer-name', ref_layer_names[0])
    for idx in range(len(mean_size)):
        channel_xml = SubElement(pre_process, 'channel')
        channel_xml.set('id', str(idx))
        mean_xml = SubElement(channel_xml, 'mean')
        mean_xml.set('offset', str(mean_offset[idx]))
        mean_xml.set('size', str(mean_size[idx]))


def create_pre_process_block(net, ref_layer_name, means, scales=None):

    pre_process = SubElement(net, 'pre-process')
    pre_process.set('reference-layer-name', ref_layer_name)

    for idx in range(len(means)):
        channel_xml = SubElement(pre_process, 'channel')
        channel_xml.set('id', str(idx))

        mean_xml = SubElement(channel_xml, 'mean')
        mean_xml.set('value', str(means[idx]))

        if scales:
            scale_xml = SubElement(channel_xml, 'scale')
            scale_xml.set('value', str(scales[idx]))

    return pre_process


def add_meta_data(net: xml.etree.ElementTree.Element, meta_info: dict):
    meta = SubElement(net, 'meta_data')
    SubElement(meta, 'MO_version').set('value', get_version())
    parameters = SubElement(meta, 'cli_parameters')
    [SubElement(parameters, str(key)).set('value', str(meta_info[key])) for key in sorted(meta_info.keys()) if
     key != 'unset']
    SubElement(parameters, 'unset').set(
        'unset_cli_parameters', ', '.join(sorted(meta_info['unset'])))


def serialize_network(graph, net_element, unsupported):
    layers = SubElement(net_element, 'layers')
    edges = SubElement(net_element, 'edges')
    if graph is None:
        return
    nodes = sorted(graph.nodes())
    for node in nodes:
        node = Node(graph, node)
        if not node.has('IE'):
            continue
        if node.kind == 'op' and (not node.has('type') or node.type is None):
            unsupported.add(node)
            continue
        try:
            serialize_node_attributes(
                graph, node, node.IE, layers, edges, unsupported)
        except Error as e:
            raise Error(str(e).replace(
                '<SUB-ELEMENT>', '{} (id = {})'.format(node.soft_get('name'), node.id))) from e


def generate_ie_ir(graph: nx.MultiDiGraph, file_name: str, input_names: tuple = (), mean_offset: tuple = (),
                   mean_size: tuple = (), meta_info: dict = dict()):

    net = Element('net')
    net.set('name', graph.name)
    net.set('version', str((graph.graph['ir_version'])))
    # TODO substitute real batches here (is it a number or is it an index?)
    net.set('batch', '1')

    if mean_size or mean_offset:
        create_pre_process_block_for_image(
            net, input_names, mean_offset, mean_size)

    if 'mean_values' in graph.graph.keys():
        for input_name, values in graph.graph['mean_values'].items():
            create_pre_process_block(net, input_name, values)

    unsupported = UnsupportedOps(graph)

    serialize_network(graph, net, unsupported)
    add_meta_data(net, meta_info)
    xml_string = tostring(net)
    xml_doc = xml.dom.minidom.parseString(xml_string)  # ugly?
    pretty_xml_as_string = xml_doc.toprettyxml()
    if len(unsupported.unsupported):
        log.debug('Partially correct IR XML:\n{}'.format(pretty_xml_as_string))
        raise Error('Part of the nodes was not translated to IE. Stopped. ' +
                    refer_to_faq_msg(24))

    return xml_string


def port_renumber(graph: nx.MultiDiGraph):
    for node in list(graph.nodes()):
        node = Node(graph, node)
        if node.kind == 'op':
            base = 0
            for u, d in sorted_inputs(node):
                d['in'] = base
                base += 1
            for v, d in sorted_outputs(node):
                d['out'] = base
                base += 1

from enum import Enum
import numpy as np
import onnx
from onnx import helper, numpy_helper

class NodeFactory:
    node_count_ = 0
    const_count_ = 0

    def __init__(self, main_graph, sub_graph=None):
        self.graph_ = sub_graph if sub_graph else main_graph
        self.main_graph_ = main_graph
        self.name_prefix_ = ''

    class ScopedPrefix:
        def __init__(self, nf, name):
            self.name_ = name
            self.nf_ = nf

        def __enter__(self):
            self.saved_name_ = self.nf_.name_prefix_
            self.nf_.name_prefix_ = self.name_

        def __exit__(self, type, value, tb):
            self.nf_.name_prefix_ = self.saved_name_

    def scoped_prefix(self, prefix):
        return NodeFactory.ScopedPrefix(self, prefix)

    def get_prefix(self, prefix):
        return self.name_prefix_

    def get_initializer(self, name):
        found = [i for i in list(self.main_graph_.initializer) + list(self.graph_.initializer) if i.name == name]
        if found:
            return numpy_helper.to_array(found[0])
        return None

    def get_value_info(self, name):
        found = [vi for vi in list(self.graph_.value_info) + list(self.graph_.input) if vi.name == name]
        if found:
            return found[0]
        return None

    def remove_initializer(self, name, allow_empty=False):
        removed = False
        for graph in [self.main_graph_, self.graph_] if self.main_graph_ != self.graph_ else [self.main_graph_]:
            initializer = [i for i in graph.initializer if i.name == name]
            if not initializer:
                continue
            assert not removed
            graph.initializer.remove(initializer[0])
            initializer_in_input = [i for i in graph.input if i.name == name]
            if initializer_in_input:
                graph.input.remove(initializer_in_input[0])
            removed = True
        assert removed or allow_empty

    @staticmethod
    def get_attribute(node, attr_name, default_value=None):
        found = [attr for attr in node.attribute if attr.name == attr_name]
        if found:
            return helper.get_attribute_value(found[0])
        return default_value

    class ValueInfoType(Enum):
        input = 1
        output = 2

    def make_value_info(self, node_or_name, data_type, shape=None, usage=None):
        if usage == NodeFactory.ValueInfoType.input:
            value_info = self.graph_.input.add()
        elif usage == NodeFactory.ValueInfoType.output:
            value_info = self.graph_.output.add()
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

    def make_initializer(self, ndarray, name='', in_main_graph=False):
        new_initializer = (self.main_graph_ if in_main_graph else self.graph_).initializer.add()
        new_name = name
        if len(new_name) == 0:
            already_existed = True
            while already_existed:
                new_name = self.name_prefix_ + '_Const_' + str(NodeFactory.const_count_)
                NodeFactory.const_count_ = NodeFactory.const_count_ + 1
                already_existed = new_name in [i.name for i in list(self.main_graph_.initializer) + list(self.graph_.initializer)]
        new_initializer.CopyFrom(numpy_helper.from_array(ndarray, new_name))
        return new_initializer

    def make_node(self, op_type, inputs, attributes={}, output_names=None, node=None):
        if type(inputs) != list:
            inputs = [inputs]
        if output_names and type(output_names) != list:
            output_names = [output_names]
        input_names = []
        for i in inputs:
            if type(i) == onnx.NodeProto or type(i) == onnx.TensorProto:
                input_names.append(i.name)
            elif type(i) == str:
                input_names.append(i)
            elif type(i) == np.ndarray:
                new_initializer = self.make_initializer(i)
                input_names.append(new_initializer.name)
            else:
                assert False # unexpected type in input

        if not node:
            node = self.graph_.node.add()

        name = self.name_prefix_ + op_type + '_' + str(NodeFactory.node_count_)
        NodeFactory.node_count_ = NodeFactory.node_count_ + 1

        if not output_names:
            output_names = [name]

        node.CopyFrom(helper.make_node(op_type, input_names, output_names, name, **attributes))
        return node

# Old models (ir_version < 4) is required to initializers in graph inputs
# This is optional for ir_version >= 4
def remove_initializers_from_inputs(mp):
    if mp.ir_version < 4:
        return

    def _append_initializer_from_graph(graph):
        initializers = [i.name for i in graph.initializer]
        for node in graph.node:
            if node.op_type == 'Scan': # currently only handle Scan
                body = [attr for attr in in_n.attribute if attr.name == 'body'][0]
                initializers += _append_initializer_from_graph(body.g)
        return initializers

    all_initializer_names = _append_initializer_from_graph(mp.graph)
    new_inputs = [vi for vi in mp.graph.input if not vi.name in all_initializer_names]
    mp.graph.ClearField('input')
    mp.graph.input.extend(new_inputs)
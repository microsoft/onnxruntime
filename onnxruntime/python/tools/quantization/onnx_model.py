import onnx
from .quant_utils import find_by_name
from pathlib import Path


class ONNXModel:
    def __init__(self, model):
        self.model = model
        self.node_name_counter = {}

    def nodes(self):
        return self.model.graph.node

    def initializer(self):
        return self.model.graph.initializer

    def graph(self):
        return self.model.graph

    def ir_version(self):
        return self.model.ir_version

    def opset_import(self):
        return self.model.opset_import

    def remove_node(self, node):
        if node in self.model.graph.node:
            self.model.graph.node.remove(node)

    def remove_nodes(self, nodes_to_remove):
        for node in nodes_to_remove:
            self.remove_node(node)

    def add_node(self, node):
        self.model.graph.node.extend([node])

    def add_nodes(self, nodes_to_add):
        self.model.graph.node.extend(nodes_to_add)

    def add_initializer(self, tensor):
        if find_by_name(tensor.name, self.model.graph.initializer) is None:
            self.model.graph.initializer.extend([tensor])

    def get_initializer(self, name):
        for tensor in self.model.graph.initializer:
            if tensor.name == name:
                return tensor
        return None

    def remove_initializer(self, tensor):
        if tensor in self.model.graph.initializer:
            self.model.graph.initializer.remove(tensor)
            for input in self.model.graph.input:
                if input.name == tensor.name:
                    self.model.graph.input.remove(input)
                    break

    def remove_initializers(self, init_to_remove):
        for initializer in init_to_remove:
            self.remove_initializer(initializer)

    def input_name_to_nodes(self):
        input_name_to_nodes = {}
        for node in self.model.graph.node:
            for input_name in node.input:
                if input_name not in input_name_to_nodes:
                    input_name_to_nodes[input_name] = [node]
                else:
                    input_name_to_nodes[input_name].append(node)
        return input_name_to_nodes

    def output_name_to_node(self):
        output_name_to_node = {}
        for node in self.model.graph.node:
            for output_name in node.output:
                output_name_to_node[output_name] = node
        return output_name_to_node

    def get_children(self, node, input_name_to_nodes=None):
        if input_name_to_nodes is None:
            input_name_to_nodes = self.input_name_to_nodes()

        children = []
        for output in node.output:
            if output in input_name_to_nodes:
                for node in input_name_to_nodes[output]:
                    children.append(node)
        return children

    def get_parents(self, node, output_name_to_node=None):
        if output_name_to_node is None:
            output_name_to_node = self.output_name_to_node()

        parents = []
        for input in node.input:
            if input in output_name_to_node:
                parents.append(output_name_to_node[input])
        return parents

    def get_parent(self, node, idx, output_name_to_node=None):
        if output_name_to_node is None:
            output_name_to_node = self.output_name_to_node()

        if len(node.input) <= idx:
            return None

        input = node.input[idx]
        if input not in output_name_to_node:
            return None

        return output_name_to_node[input]

    def find_node_by_name(self, node_name, new_nodes_list, graph):
        '''
        Find out if a node exists in a graph or a node is in the 
        new set of nodes created during quantization. Return the node found.
        '''
        graph_nodes_list = list(graph.node)  #deep copy
        graph_nodes_list.extend(new_nodes_list)
        node = find_by_name(node_name, graph_nodes_list)
        return node

    def find_nodes_by_initializer(self, graph, initializer):
        '''
        Find all nodes with given initializer as an input.
        '''
        nodes = []
        for node in graph.node:
            for node_input in node.input:
                if node_input == initializer.name:
                    nodes.append(node)
        return nodes

    def replace_gemm_with_matmul(self):
        new_nodes = []

        for node in self.nodes():
            if node.op_type == 'Gemm':
                alpha = 1.0
                beta = 1.0
                transA = 0
                transB = 0
                for attr in node.attribute:
                    if attr.name == 'alpha':
                        alpha = onnx.helper.get_attribute_value(attr)
                    elif attr.name == 'beta':
                        beta = onnx.helper.get_attribute_value(attr)
                    elif attr.name == 'transA':
                        transA = onnx.helper.get_attribute_value(attr)
                    elif attr.name == 'transB':
                        transB = onnx.helper.get_attribute_value(attr)
                if alpha == 1.0 and beta == 1.0 and transA == 0:
                    inputB = node.input[1]
                    if transB == 1:
                        B = self.get_initializer(node.input[1])
                        if B:
                            # assume B is not used by any other node
                            B_array = onnx.numpy_helper.to_array(B)
                            B_trans = onnx.numpy_helper.from_array(B_array.T)
                            B_trans.name = B.name
                            self.remove_initializer(B)
                            self.add_initializer(B_trans)
                        else:
                            inputB += '_Transposed'
                            transpose_node = onnx.helper.make_node('Transpose',
                                                                   inputs=[node.input[1]],
                                                                   outputs=[inputB],
                                                                   name=node.name + '_Transpose')
                            new_nodes.append(transpose_node)

                    matmul_node = onnx.helper.make_node(
                        'MatMul',
                        inputs=[node.input[0], inputB],
                        outputs=[node.output[0] + ('_MatMul' if len(node.input) > 2 else '')],
                        name=node.name + '_MatMul')
                    new_nodes.append(matmul_node)

                    if len(node.input) > 2:
                        add_node = onnx.helper.make_node('Add',
                                                         inputs=[node.output[0] + '_MatMul', node.input[2]],
                                                         outputs=node.output,
                                                         name=node.name + '_Add')
                        new_nodes.append(add_node)

                # unsupported
                else:
                    new_nodes.append(node)

            # not GEMM
            else:
                new_nodes.append(node)

        self.graph().ClearField('node')
        self.graph().node.extend(new_nodes)

    def save_model_to_file(self, output_path, use_external_data_format=False):
        '''
        Save model to external data, which is needed for model size > 2GB
        '''
        if use_external_data_format:
            onnx.external_data_helper.convert_model_to_external_data(self.model,
                                                                     all_tensors_to_one_file=True,
                                                                     location=Path(output_path).name + ".data")
        onnx.save_model(self.model, output_path)

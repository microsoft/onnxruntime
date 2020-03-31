import sys
import os
import onnx
from onnx import helper
from onnx import TensorProto
from onnx import OperatorSetIdProto

# Edge that needs to be cut for the split.
# If the edge is feeding into more than one nodes, and not all the nodes belong to the same cut,
# specify those consuming nodes that need to be cut


class CutEdge:
    def __init__(self, edgeId, consumingNodes=None):
        self.edgeId = edgeId
        self.consumingNodes = consumingNodes


def add_expand_type(model, name, type):
    expand_edge = model.graph.value_info.add()
    expand_edge.name = name
    expand_edge.type.CopyFrom(type)

# Add wait/record/send/recv nodes and split the graph into disconnected subgraphs


def split_graph(model, split_edge_groups):
    ms_domain = 'com.microsoft'

    new_send_nodes = []
    new_recv_nodes = []
    # Add wait for initial inputs. This needs to be done first before new inputs
    # are introduced from split
    initializer_lists = [a.name for a in model.graph.initializer]
    input_tensors = [
        value.name for value in model.graph.input if value.name not in initializer_lists]

    input_wait_signal = model.graph.input.add()
    input_wait_signal.CopyFrom(helper.make_tensor_value_info(
        'input_wait_signal', onnx.TensorProto.INT64, None))

    input_wait = model.graph.node.add()
    input_wait.CopyFrom(helper.make_node(
        'WaitEvent',
        inputs=['input_wait_signal'],
        outputs=[],
        domain=ms_domain))

    for i in input_tensors:
        for node in model.graph.node:
            for j in range(len(node.input)):
                if node.input[j] == i:
                    node.input[j] = i + '_sync'

    input_wait.input.extend(input_tensors)
    input_wait.output.extend([i + '_sync' for i in input_tensors])

    for cut_index in range(len(split_edge_groups)):
        edgeIds = split_edge_groups[cut_index]

        # split the graph based on edgeIds
        upstream_nodes = []
        upstream_nodes_output_index = []
        output_shapes = []

        for id in edgeIds:
            for node in model.graph.node:
                if len(node.output) >= 1:
                    for i, j in enumerate(node.output):
                        if j == id:
                            upstream_nodes.append(node)
                            upstream_nodes_output_index.append(i)
                    for info in model.graph.value_info:
                        if info.name == id:
                            output_shapes.append(info.type)

        record_signal = model.graph.input.add()
        record_signal.CopyFrom(helper.make_tensor_value_info(
            'record_input_signal' + str(cut_index), onnx.TensorProto.INT64, None))

        wait_signal = model.graph.input.add()
        wait_signal.CopyFrom(helper.make_tensor_value_info(
            'wait_input_signal' + str(cut_index), onnx.TensorProto.INT64, None))

        send_signal = model.graph.input.add()
        send_signal.CopyFrom(helper.make_tensor_value_info(
            'send_input_signal' + str(cut_index), onnx.TensorProto.BOOL, None))

        recv_signal = model.graph.input.add()
        recv_signal.CopyFrom(helper.make_tensor_value_info(
            'recv_input_signal' + str(cut_index), onnx.TensorProto.BOOL, None))

        # output signal from send after cut
        send_output_signal = model.graph.output.add()
        send_output_signal.CopyFrom(helper.make_tensor_value_info(
            'send_output_signal' + str(cut_index), onnx.TensorProto.BOOL, None))

        # output signal from receive after cut
        receive_output_signal = model.graph.output.add()
        receive_output_signal.CopyFrom(helper.make_tensor_value_info(
            'receive_output_signal' + str(cut_index), onnx.TensorProto.BOOL, None))

        new_send = model.graph.node.add()
        new_send.CopyFrom(helper.make_node(
            'Send',
            inputs=['send_input_signal' + str(cut_index)],
            outputs=['send_output_signal' + str(cut_index)],
            tag=0,
            src=cut_index,
            dst=cut_index + 1,
            domain=ms_domain,
            element_type=7,  # assuming all tensors are of type float
            name='send'))

        new_receive = model.graph.node.add()
        new_receive.CopyFrom(helper.make_node(
            'Recv',
            inputs=['recv_input_signal' + str(cut_index)],
            outputs=['receive_output_signal' + str(cut_index)],
            tag=1,
            src=cut_index,
            dst=cut_index + 1,
            domain=ms_domain,
            element_type=7,  # assuming all tensors are of type float
            name='receive'))

        new_wait = model.graph.node.add()
        new_wait.CopyFrom(helper.make_node(
            'WaitEvent',
            inputs=['wait_input_signal' + str(cut_index)],
            outputs=[],
            domain=ms_domain))

        new_record = model.graph.node.add()
        new_record.CopyFrom(helper.make_node(
            'RecordEvent',
            inputs=['record_input_signal' + str(cut_index)],
            outputs=[],
            domain=ms_domain))

        for i in range(len(upstream_nodes)):
            n = upstream_nodes[i]
            idx = upstream_nodes_output_index[i]
            output_type = output_shapes[i]
            output_edge_name = n.output[idx]

            output_nodes = find_all_output_nodes_by_edge(
                model, output_edge_name)

            # deal with shape inference for newly added edge
            new_send_input_name = output_edge_name + '_send' + str(cut_index)
            add_expand_type(model, new_send_input_name, output_type)

            new_receive_output_name = output_edge_name + \
                '_recv' + str(cut_index)
            add_expand_type(model, new_receive_output_name, output_type)

            new_wait_output_name = output_edge_name + '_wait' + str(cut_index)
            add_expand_type(model, new_wait_output_name, output_type)

            # the order of data flow is: node-output -> record -> send -> recv -> wait -> node-input
            new_record.input.extend([output_edge_name])
            new_record.output.extend([new_send_input_name])

            new_send.input.extend([new_send_input_name])
            new_receive.output.extend([new_receive_output_name])

            new_wait.input.extend([new_receive_output_name])
            new_wait.output.extend([new_wait_output_name])

            for output_node in output_nodes:
                for i in range(len(output_node.input)):
                    for edgeId in edgeIds:
                        if output_node.input[i] == edgeId:
                            output_node.input[i] = new_wait_output_name

        new_send_nodes.append(new_send)
        new_recv_nodes.append(new_receive)

    model = onnx.shape_inference.infer_shapes(model)

    return new_send_nodes, new_recv_nodes


def find_all_input_nodes(model, node):
    nodes = []
    inputs = []

    if node:
        for inputId in node.input:
            nodes.extend([n for n in model.graph.node if inputId in n.output])
            inputs.extend([n for n in model.graph.input if inputId in n.name])
    return nodes, inputs


def find_all_output_nodes(model, node):
    nodes = []
    outputs = []
    if node:
        for outputId in node.output:
            nodes.extend([n for n in model.graph.node if outputId in n.input])
            outputs.extend(
                [n for n in model.graph.output if outputId in n.name])
    return nodes, outputs


def find_all_output_nodes_by_edge(model, arg):
    result = [n for n in model.graph.node if arg in n.input]
    return result

# Insert identity nodes to separate same output edge which feeds into different sub-graph.


def add_identity(model, cuttingEdge, newEdgeIdName):
    output_nodes = None
    edgeId = cuttingEdge.edgeId
    for node in model.graph.node:
        if len(node.output) >= 1:
            for output in node.output:
                if output == edgeId:
                    output_nodes = find_all_output_nodes_by_edge(model, output)
                    break

    assert output_nodes, "no output node"

    new_identity = model.graph.node.add()
    new_identity.op_type = 'Identity'

    new_identity.input.extend([edgeId])
    new_identity.output.extend([newEdgeIdName])

    for i in range(len(output_nodes)):
        for output in output_nodes[i].output:
            if output in cuttingEdge.consumingNodes:
                for j in range(len(output_nodes[i].input)):
                    if output_nodes[i].input[j] == edgeId:
                        output_nodes[i].input[j] = newEdgeIdName

    return newEdgeIdName


def find_all_connected_nodes(model, node):
    nodes0, inputs = find_all_input_nodes(model, node)
    nodes1, outputs = find_all_output_nodes(model, node)

    connected_nodes = nodes0 + nodes1
    return connected_nodes, inputs, outputs


def get_node_index(model, node):
    found = [i for i, n in enumerate(model.graph.node) if n == node]
    return found[0] if found else None


def get_input_index(model, input):
    found = [i for i, n in enumerate(model.graph.input) if n == input]
    return found[0] if found else None


def get_output_index(model, output):
    found = [i for i, n in enumerate(model.graph.output) if n == output]
    return found[0] if found else None

# traverse the graph, group connected nodes and generate subgraph


def generate_subgraph(model, start_nodes):
    subgraphs = []

    main_graph = onnx.ModelProto()
    main_graph.CopyFrom(model)

    all_visited_nodes = []
    model_count = len(start_nodes)
    for start in reversed(start_nodes):
        stack0 = [start]

        visited0 = []
        tranversed_node = 0
        inputs0 = []
        outputs0 = []
        while stack0:
            node = stack0.pop()
            if not node in visited0:
                tranversed_node += 1
                visited0.append(node)
                all_visited_nodes.append(node)
                connected_nodes, inputs, outputs = find_all_connected_nodes(
                    main_graph, node)

                stack0 = stack0 + connected_nodes
                inputs0 = inputs0 + inputs
                outputs0 = outputs0 + outputs

        subgraph = onnx.ModelProto()
        subgraph.CopyFrom(main_graph)

        # gather visited nodes
        visited_nodes = []
        for n in visited0:
            visited_nodes.append(get_node_index(main_graph, n))
        visited_nodes.sort(reverse=True)

        # gather visited inputs
        visited_inputs = []
        for n in inputs0:
            visited_inputs.append(get_input_index(main_graph, n))
        visited_inputs.sort(reverse=True)

        # gather visited outputs
        visited_outputs = []
        for n in outputs0:
            visited_outputs.append(get_output_index(main_graph, n))
        visited_outputs.sort(reverse=True)

        for i in reversed(range(len(main_graph.graph.node))):
            try:
                if i not in visited_nodes:
                    del subgraph.graph.node[i]
                else:
                    del main_graph.graph.node[i]
            except:
                print("error deleting node", i)

        for i in reversed(range(len(main_graph.graph.input))):
            try:
                if i not in visited_inputs:
                    del subgraph.graph.input[i]
                else:
                    del main_graph.graph.input[i]
            except:
                print("error deleting inputs", i)

        for i in reversed(range(len(main_graph.graph.output))):
            try:
                if i not in visited_outputs:
                    del subgraph.graph.output[i]
                else:
                    del main_graph.graph.output[i]
            except:
                print("error deleting outputs ", i)

        print("model", str(model_count), " length ", len(subgraph.graph.node))
        subgraphs.append(subgraph)
        model_count -= 1

    print("model", str(model_count), " length ", len(main_graph.graph.node))
    subgraphs.append(main_graph)

    # as the subgraphs were added in reverse order (the last split is added first), reverse the order back before return
    subgraphs.reverse()
    return subgraphs


def write_model(model, file_name):
    onnx.save(model, file_name)


def main():
    # temporary hard coded the cutting edge structure
    # TODO: move this info to a file (json?) and load the data from there.
    input_model_name = 'bert-tiny-uncased_L_3_H_128_A_2_V_30528_S_512_Dp_0.1.onnx'
    stage_count = 3

    cut0_input = {CutEdge('186'), CutEdge('71', {'273', '395'})}
    cut1_input = {CutEdge('308'), CutEdge('71', {'395'})}
    all_cut_inputs = [cut0_input, cut1_input]

    model = onnx.load(input_model_name)
    if len(model.graph.value_info) == 0:
        model = onnx.shape_inference.infer_shapes(model)

    print("original model length ", len(model.graph.node))

    output_model_names = [os.path.splitext(input_model_name)[0] + '_' +
                          str(i) + '.onnx' for i in range(stage_count)]

    split_edge_groups = []
    count = 0
    updated_edges = {}
    need_shape_inference = False
    # Sweep the cut edge to see if there are edges feeding into nodes from two sub-graphs. If so,
    # insert identity node after those edges with a new ID to distinguish the rest.
    for cut_input in all_cut_inputs:
        split_edges = []
        for i in cut_input:
            if i.consumingNodes:
                # if this edge has previously been modified, update its edgeId before inserting new identity
                if i.edgeId in updated_edges:
                    i.edgeId = updated_edges[i.edgeId]

                new_edge_name = 'identity_output_' + str(count)
                add_identity(model, i, new_edge_name)
                count += 1
                split_edges.append(new_edge_name)
                updated_edges[i.edgeId] = new_edge_name
                need_shape_inference = True
            else:
                split_edges.append(i.edgeId)
        split_edge_groups.append(split_edges)

    # new edge is being added, need to re-inference shape
    if need_shape_inference:
        model = onnx.shape_inference.infer_shapes(model)

    # after all need-to-be-cut edges identified, split the graph
    new_sends, new_receives = split_graph(model, split_edge_groups)
    sub_graphs = generate_subgraph(model, new_receives)

    for i in range(stage_count):
        sub_graphs[i] = onnx.shape_inference.infer_shapes(sub_graphs[i])
        write_model(sub_graphs[i], output_model_names[i])


if __name__ == "__main__":
    main()

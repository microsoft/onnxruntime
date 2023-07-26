import os
import sys  # noqa: F401

import onnx
from onnx import OperatorSetIdProto, TensorProto, helper  # noqa: F401

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
    ms_domain = "com.microsoft"

    new_send_nodes = []
    new_recv_nodes = []

    for cut_index in range(len(split_edge_groups)):
        edgeIds = split_edge_groups[cut_index]  # noqa: N806

        # split the graph based on edgeIds
        upstream_nodes = []
        upstream_nodes_output_index = []
        output_shapes = []
        element_types = []
        for id in edgeIds:
            for node in model.graph.node:
                if len(node.output) >= 1:
                    for i, j in enumerate(node.output):
                        if j == id:
                            upstream_nodes.append(node)
                            upstream_nodes_output_index.append(i)
                            # assuming all tensors are of type float
                            element_types.append(1)
            for info in model.graph.value_info:
                if info.name == id:
                    output_shapes.append(info.type)

        send_input_signal_name = "send_input_signal" + str(cut_index)
        send_signal = model.graph.input.add()
        send_signal.CopyFrom(helper.make_tensor_value_info(send_input_signal_name, onnx.TensorProto.BOOL, None))
        send_signal = helper.make_tensor(send_input_signal_name, TensorProto.BOOL, (), (True,))
        model.graph.initializer.extend([send_signal])

        recv_input_signal_name = "recv_input_signal" + str(cut_index)
        recv_signal = model.graph.input.add()
        recv_signal.CopyFrom(helper.make_tensor_value_info(recv_input_signal_name, onnx.TensorProto.BOOL, None))
        recv_signal = helper.make_tensor(recv_input_signal_name, TensorProto.BOOL, (), (True,))
        model.graph.initializer.extend([recv_signal])

        send_dst_rank_name = "send_dst_rank" + str(cut_index)
        send_dst_rank = model.graph.input.add()
        send_dst_rank.CopyFrom(helper.make_tensor_value_info(send_dst_rank_name, onnx.TensorProto.INT64, None))
        send_dst_rank = helper.make_tensor(send_dst_rank_name, TensorProto.INT64, (), (cut_index + 1,))
        model.graph.initializer.extend([send_dst_rank])

        recv_src_rank_name = "recv_src_rank" + str(cut_index)
        recv_src_rank = model.graph.input.add()
        recv_src_rank.CopyFrom(helper.make_tensor_value_info(recv_src_rank_name, onnx.TensorProto.INT64, None))
        recv_src_rank = helper.make_tensor(recv_src_rank_name, TensorProto.INT64, (), (cut_index,))
        model.graph.initializer.extend([recv_src_rank])

        # output signal from send after cut
        send_output_signal = model.graph.output.add()
        send_output_signal.CopyFrom(
            helper.make_tensor_value_info("send_output_signal" + str(cut_index), onnx.TensorProto.BOOL, None)
        )

        # output signal from receive after cut
        receive_output_signal = model.graph.output.add()
        receive_output_signal.CopyFrom(
            helper.make_tensor_value_info("receive_output_signal" + str(cut_index), onnx.TensorProto.BOOL, None)
        )

        new_send = model.graph.node.add()
        new_send.CopyFrom(
            helper.make_node(
                "Send",
                inputs=[send_input_signal_name, send_dst_rank_name],
                outputs=["send_output_signal" + str(cut_index)],
                tag=0,
                domain=ms_domain,
                element_types=element_types,
                name="send",
            )
        )

        new_receive = model.graph.node.add()
        new_receive.CopyFrom(
            helper.make_node(
                "Recv",
                inputs=[recv_input_signal_name, recv_src_rank_name],
                outputs=["receive_output_signal" + str(cut_index)],
                tag=0,
                domain=ms_domain,
                element_types=element_types,
                name="receive",
            )
        )

        for i in range(len(upstream_nodes)):
            n = upstream_nodes[i]
            idx = upstream_nodes_output_index[i]
            output_type = output_shapes[i]
            output_edge_name = n.output[idx]

            output_nodes = find_all_output_nodes_by_edge(model, output_edge_name)

            # deal with shape inference for newly added edge
            new_send_input_name = output_edge_name + "_send" + str(cut_index)
            add_expand_type(model, new_send_input_name, output_type)

            new_receive_output_name = output_edge_name + "_recv" + str(cut_index)
            add_expand_type(model, new_receive_output_name, output_type)

            # the order of data flow is: node-output -> record -> send -> recv -> wait -> node-input

            new_send.input.extend([output_edge_name])
            new_receive.output.extend([new_receive_output_name])

            for output_node in output_nodes:
                for i in range(len(output_node.input)):  # noqa: PLW2901
                    for edgeId in edgeIds:  # noqa: N806
                        if output_node.input[i] == edgeId:
                            output_node.input[i] = new_receive_output_name

        new_send_nodes.append(new_send)
        new_recv_nodes.append(new_receive)

    model = onnx.shape_inference.infer_shapes(model)

    return new_send_nodes, new_recv_nodes


def find_all_input_nodes(model, node):
    nodes = []
    inputs = []

    if node:
        for inputId in node.input:  # noqa: N806
            nodes.extend([n for n in model.graph.node if inputId in n.output])
            inputs.extend([n for n in model.graph.input if inputId in n.name])
    return nodes, inputs


def find_all_output_nodes(model, node):
    nodes = []
    outputs = []
    if node:
        for outputId in node.output:  # noqa: N806
            nodes.extend([n for n in model.graph.node if outputId in n.input])
            outputs.extend([n for n in model.graph.output if outputId in n.name])
    return nodes, outputs


def find_all_output_nodes_by_edge(model, arg):
    result = [n for n in model.graph.node if arg in n.input]
    return result


# Insert identity nodes to separate same output edge which feeds into different sub-graph.


def add_identity(model, cuttingEdge, newEdgeIdName):
    output_nodes = None
    edgeId = cuttingEdge.edgeId  # noqa: N806
    for node in model.graph.node:
        if len(node.output) >= 1:
            for output in node.output:
                if output == edgeId:
                    output_nodes = find_all_output_nodes_by_edge(model, output)
                    break

    assert output_nodes, "no output node"

    new_identity = model.graph.node.add()
    new_identity.op_type = "Identity"

    new_identity.input.extend([edgeId])
    new_identity.output.extend([newEdgeIdName])

    for i in range(len(output_nodes)):
        for output in output_nodes[i].output:
            if output in cuttingEdge.consumingNodes:
                for j in range(len(output_nodes[i].input)):
                    if output_nodes[i].input[j] == edgeId:
                        output_nodes[i].input[j] = newEdgeIdName

    return new_identity


def insert_identity(model, all_cut_inputs):
    count = 0
    updated_edges = {}
    new_added_identity = []
    split_edge_groups = []
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

                new_edge_name = "identity_output_" + str(count)
                new_added_identity.append(add_identity(model, i, new_edge_name))
                count += 1
                split_edges.append(new_edge_name)
                updated_edges[i.edgeId] = new_edge_name
                need_shape_inference = True
            else:
                split_edges.append(i.edgeId)
        split_edge_groups.append(split_edges)
    return split_edge_groups, new_added_identity, need_shape_inference


# after the graph is split, remove the added identity node because identity op is not registered in gradient builder.


def remove_identity(model, new_added_identity):
    for node in new_added_identity:
        assert node.op_type == "Identity"
        output_nodes = [n for n in model.graph.node if node.output[0] in n.input]
        for output_node in output_nodes:
            for i in range(len(output_node.input)):
                if output_node.input[i] == node.output[0]:
                    output_node.input[i] = node.input[0]


def find_all_connected_nodes(model, node):
    nodes0, inputs = find_all_input_nodes(model, node)
    nodes1, outputs = find_all_output_nodes(model, node)

    connected_nodes = nodes0 + nodes1
    return connected_nodes, inputs, outputs


def get_index(node_list, node):
    found = [i for i, n in enumerate(node_list) if n == node]
    return found[0] if found else None


def get_identity_index_for_deleting(node_list, node):
    for i, n in enumerate(node_list):
        # The node's input name has been changed during send/recv insertion,
        # but it is sufficient to just compare the type and outputs.
        if n.op_type == "Identity" and n.output == node.output:
            return i
    return None


# traverse the graph, group connected nodes and generate subgraph


def generate_subgraph(model, start_nodes, identity_node_list):
    subgraphs = []

    main_graph = onnx.ModelProto()
    main_graph.CopyFrom(model)

    # remove added identity node before copy to subgraph
    identity_node_index = []
    for n in identity_node_list:
        identity_node_index.append(get_identity_index_for_deleting(main_graph.graph.node, n))
    identity_node_index.sort(reverse=True)

    for i in reversed(range(len(main_graph.graph.node))):
        try:
            if i in identity_node_index:
                del main_graph.graph.node[i]
        except Exception:
            print("error deleting identity node", i)

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
            if node not in visited0:
                tranversed_node += 1
                visited0.append(node)
                all_visited_nodes.append(node)
                connected_nodes, inputs, outputs = find_all_connected_nodes(main_graph, node)

                stack0 = stack0 + connected_nodes
                inputs0 = inputs0 + inputs
                outputs0 = outputs0 + outputs

        subgraph = onnx.ModelProto()
        subgraph.CopyFrom(main_graph)

        # gather visited nodes
        visited_nodes = []
        for n in visited0:
            visited_nodes.append(get_index(main_graph.graph.node, n))
        visited_nodes.sort(reverse=True)

        # gather visited inputs
        visited_inputs = []
        for n in inputs0:
            visited_inputs.append(get_index(main_graph.graph.input, n))
        visited_inputs.sort(reverse=True)

        # gather visited outputs
        visited_outputs = []
        for n in outputs0:
            visited_outputs.append(get_index(main_graph.graph.output, n))
        visited_outputs.sort(reverse=True)

        for i in reversed(range(len(main_graph.graph.node))):
            try:
                if i not in visited_nodes:
                    del subgraph.graph.node[i]
                else:
                    del main_graph.graph.node[i]
            except Exception:
                print("error deleting node", i)

        for i in reversed(range(len(main_graph.graph.input))):
            try:
                if i not in visited_inputs:
                    del subgraph.graph.input[i]
                else:
                    del main_graph.graph.input[i]
            except Exception:
                print("error deleting inputs", i)

        for i in reversed(range(len(main_graph.graph.output))):
            try:
                if i not in visited_outputs:
                    del subgraph.graph.output[i]
                else:
                    del main_graph.graph.output[i]
            except Exception:
                print("error deleting outputs ", i)

        print("model", str(model_count), " length ", len(subgraph.graph.node))
        subgraphs.append(subgraph)
        model_count -= 1

    print("model", str(model_count), " length ", len(main_graph.graph.node))
    subgraphs.append(main_graph)

    # as the subgraphs were added in reverse order (the last split is added first), reverse the order back before return
    subgraphs.reverse()
    return subgraphs


def main():
    # temporary hard coded the cutting edge structure
    # TODO: move this info to a file (json?) and load the data from there.
    input_model_name = "bert-tiny-uncased_L_3_H_128_A_2_V_30528_S_512_Dp_0.1.onnx"
    stage_count = 3

    cut0_input = {CutEdge("186"), CutEdge("71", {"273", "395"})}
    cut1_input = {CutEdge("308"), CutEdge("71", {"395"})}
    all_cut_inputs = [cut0_input, cut1_input]

    model = onnx.load(input_model_name)
    if len(model.graph.value_info) == 0:
        model = onnx.shape_inference.infer_shapes(model)

    print("original model length ", len(model.graph.node))

    output_model_names = [os.path.splitext(input_model_name)[0] + "_" + str(i) + ".onnx" for i in range(stage_count)]

    split_edge_groups, new_identity, need_shape_inference = insert_identity(model, all_cut_inputs)

    # new edge is being added, need to re-inference shape
    if need_shape_inference:
        model = onnx.shape_inference.infer_shapes(model)

    # after all need-to-be-cut edges identified, split the graph
    new_sends, new_receives = split_graph(model, split_edge_groups)
    remove_identity(model, new_identity)
    sub_graphs = generate_subgraph(model, new_receives, new_identity)

    for i in range(stage_count):
        sub_graphs[i] = onnx.shape_inference.infer_shapes(sub_graphs[i])
        onnx.save(sub_graphs[i], output_model_names[i])
        print("save to file: ", output_model_names[i])


if __name__ == "__main__":
    main()

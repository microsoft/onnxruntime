import sys
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

def add_expand_shape(model, name, shape):
    expand_edge = model.graph.value_info.add()
    expand_edge.name = name
    expand_edge.type.CopyFrom(shape)

# Add wait/record/send/recv nodes and split the graph into disconnected subgraphs
def split_graph(model, edgeIds):
    ms_domain = 'com.microsoft'

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
                    print(node.input[j])
                    node.input[j] = i + '_sync'

    input_wait.input.extend(input_tensors)
    input_wait.output.extend([i + '_sync' for i in input_tensors])

    # split the graph based on edgeIds
    upstream_nodes = []
    output_shapes =[]

    for id in edgeIds:
        for node in model.graph.node:
            if len(node.output) >=1 and node.output[0] == id:
                upstream_nodes.append(node)
                for info in model.graph.value_info:
                    if info.name == id:
                        output_shapes.append(info.type)

    record_signal = model.graph.input.add()
    record_signal.CopyFrom(helper.make_tensor_value_info(
        'record_signal', onnx.TensorProto.INT64, None))

    wait_signal = model.graph.input.add()
    wait_signal.CopyFrom(helper.make_tensor_value_info(
        'wait_signal', onnx.TensorProto.INT64, None))

    send_signal = model.graph.input.add()
    send_signal.CopyFrom(helper.make_tensor_value_info(
        'send_signal', onnx.TensorProto.BOOL, None))

    recv_signal = model.graph.input.add()
    recv_signal.CopyFrom(helper.make_tensor_value_info(
        'recv_signal', onnx.TensorProto.BOOL, None))

    # output signal from send after cut
    send_output_signal = model.graph.output.add()
    send_output_signal.CopyFrom(helper.make_tensor_value_info(
        'send_output_signal', onnx.TensorProto.BOOL, None))

    # output signal from receive after cut
    receive_output_signal = model.graph.output.add()
    receive_output_signal.CopyFrom(helper.make_tensor_value_info(
        'receive_output_signal', onnx.TensorProto.BOOL, None))

    new_send = model.graph.node.add()
    new_send.CopyFrom(helper.make_node(
        'Send',
        inputs=['send_signal'],
        outputs=['send_output_signal'],
        tag=0,
        src=0, # todo：change it when split more than 2 parts
        dst=1,
        domain=ms_domain,
        element_type=1, # assuming all tensors are of type float
        name='send'))

    new_receive = model.graph.node.add()
    new_receive.CopyFrom(helper.make_node(
        'Recv',
        inputs=['recv_signal'],
        outputs=['receive_output_signal'],
        tag=1,
        src=0, # todo：change it when split more than 2 parts
        dst=1,
        domain=ms_domain,
        element_type=1, # assuming all tensors are of type float
        name='receive'))

    new_wait = model.graph.node.add()
    new_wait.CopyFrom(helper.make_node(
        'WaitEvent',
        inputs=['wait_signal'],
        outputs=[],
        domain=ms_domain))

    new_record = model.graph.node.add()
    new_record.CopyFrom(helper.make_node(
        'RecordEvent',
        inputs=['record_signal'],
        outputs=[],
        domain=ms_domain))

    for i in range(len(upstream_nodes)):
        n = upstream_nodes[i]
        output_type = output_shapes[i]

        output_nodes = find_all_output_nodes_by_edge(model, n.output[0])

        # deal with shape inference for newly added edge
        new_send_input_name = n.output[0] + '_send'
        add_expand_shape(model, new_send_input_name, output_type)

        new_receive_output_name = n.output[0] + '_recv'
        add_expand_shape(model, new_receive_output_name, output_type)

        new_wait_output_name = n.output[0] + '_wait'
        add_expand_shape(model, new_wait_output_name, output_type)

        # the order of data flow is: node-output -> record -> send -> recv -> wait -> node-input
        new_record.input.extend([n.output[0]])
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

    return new_send, new_receive

def find_all_input_nodes(model, node):
    nodes = []
    inputs = []

    if node:
        for inputId in node.input:
            for node in model.graph.node:
                for output in node.output:
                    if output == inputId:
                        nodes.append(node)
            for input in model.graph.input:
                if input.name == inputId:
                    inputs.append(input)
    return nodes, inputs

def find_all_output_nodes(model, node):
    nodes = []
    outputs = []
    if node:
        for outputId in node.output:
            for node in model.graph.node:
                for input in node.input:
                    if input == outputId:
                        nodes.append(node)
            for output in model.graph.output:
                if output.name == outputId:
                    outputs.append(output)
    return nodes, outputs

def find_all_output_nodes_by_edge(model, arg):
    result = []
    for node in model.graph.node:
        for input in node.input:
            if input == arg:
                result.append(node)
    return result

# Insert identity nodes to separate same output edge which feeds into different sub-graph.
def add_identity(model, cuttingEdge, newEdgeIdName):
    output_nodes = None
    edgeId = cuttingEdge.edgeId
    for node in model.graph.node:
        if len(node.output) >=1 and node.output[0] == edgeId:
            output_nodes = find_all_output_nodes_by_edge(model, node.output[0])
            break

    assert output_nodes, "no output node"

    new_identity = model.graph.node.add()
    new_identity.op_type = 'Identity'

    new_identity.input.extend([edgeId])
    new_identity.output.extend([newEdgeIdName])

    for i in range(len(output_nodes)):
        if output_nodes[i].output[0] in cuttingEdge.consumingNodes:
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
    for i in range(len(model.graph.node)):
        if model.graph.node[i] == node:
            return i
    return None

def get_input_index(model, input):
    for i in range(len(model.graph.input)):
        if model.graph.input[i] == input:
            return i
    return None

def get_output_index(model, output):
    for i in range(len(model.graph.output)):
        if model.graph.output[i] == output:
            return i
    return None

# traverse the graph, group connected nodes and generate subgraph
def generate_subgraph(model, start0, start1):
    stack0 = [start0]
    stack1 = [start1]

    visited0 = []
    tranversed_node = 0
    inputs0 = []
    outputs0 = []
    while stack0:
        node = stack0.pop()
        if not node in visited0:
            tranversed_node += 1
            visited0.append(node)
            connected_nodes, inputs, outputs = find_all_connected_nodes(model, node)

            stack0 = stack0 + connected_nodes
            inputs0 = inputs0 + inputs
            outputs0 = outputs0 + outputs

    model0 = onnx.ModelProto()
    model0.CopyFrom(model)
    model1 = onnx.ModelProto()
    model1.CopyFrom(model)

    # gather deleted nodes
    delete_nodes = []
    for n in visited0:
        delete_nodes.append(get_node_index(model, n))
    delete_nodes.sort(reverse=True)

    # gather deleted inputs
    delete_inputs = []
    for n in inputs0:
        delete_inputs.append(get_input_index(model, n))
    delete_inputs.sort(reverse=True)

    # gather deleted outputs
    delete_outputs = []
    for n in outputs0:
        delete_outputs.append(get_output_index(model, n))
    delete_outputs.sort(reverse=True)

    for i in reversed(range(len(model.graph.node))):
        try:
            if i in delete_nodes:
                del model1.graph.node[i]
            else:
                del model0.graph.node[i]
        except:
            print("error deleting node", i)

    for i in reversed(range(len(model.graph.input))):
        try:
            if i in delete_inputs:
                del model1.graph.input[i]
            else:
                del model0.graph.input[i]
        except:
            print("error deleting inputs", i)

    for i in reversed(range(len(model.graph.output))):
        try:
            if i in delete_outputs:
                del model1.graph.output[i]
            else:
                del model0.graph.output[i]
        except:
            print("error deleting outputs ", i)

    print("model0 length ", len(model0.graph.node))
    print("model1 length ", len(model1.graph.node))

    return model0, model1

def write_model(model, file_name):
    f = open(file_name, "wb")
    f.write(model.SerializeToString())
    f.close()

def main():
    # temporary hard coded the cutting edge structure
    # TODO: move this info to a file (json?) and load the data from there.
    input_model_name = 'bert-tiny-uncased_L_3_H_128_A_2_V_30528_S_512_Dp_0.1.onnx'
    layer_output_edge = CutEdge('186')
    mask_output_edge = CutEdge('71', {'273', '395'})
    cut_input = {layer_output_edge, mask_output_edge}

    model = onnx.load(input_model_name)
    print("input model length: ", len(model.graph.node))

    # TODO: support split more than 2 subgraphs
    output_model_names = [input_model_name[:-5] + '_0.onnx', input_model_name[:-5] + '_1.onnx']

    split_edges = []
    count = 0
    # Sweep the cut edge to see if there are edges feeding into nodes from two sub-graphs. If so,
    # insert identity node after those edges with a new ID to distinguish the rest.
    for i in cut_input:
        if i.consumingNodes:
            new_edge_name = 'identity_output_' + str(count)
            add_identity(model, i, new_edge_name)
            count += 1
            split_edges.append(new_edge_name)
        else:
            split_edges.append(i.edgeId)

    # after all need-to-be-cut edges identified, split the graph
    new_send, new_receive = split_graph(model, split_edges)
    model0, model1 = generate_subgraph(model, new_send, new_receive)

    model0.opset_import[0].domain = "com.microsoft"
    model0.opset_import[0].version = 1

    model1.opset_import[0].domain = "com.microsoft"
    model1.opset_import[0].version = 1

    write_model(model0, output_model_names[0])
    write_model(model1, output_model_names[1])

if __name__ == "__main__":
    main()


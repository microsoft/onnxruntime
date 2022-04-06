from .graph import Graph
import onnx
import onnx
from onnx import helper
from onnx import TensorProto, OperatorSetIdProto
import copy

class MSELoss(Graph):
    def __init__(self):
        super(MSELoss, self).__init__()

    def build(self, base_model, output, target='target', reduction='mean'):
        # Ideally
        # model = onnx_model.make_functional()
        # loss_unreduced = onnx.Pow(onnx.Sub(model(), target), 2)
        # if reduction == 'mean':
        #     loss = onnx.ReduceMean(loss_unreduced)
        # elif reduction == 'sum':
        #     loss = onnx.ReduceSum(loss_unreduced)
        # return loss

        # deepcopy the base model so we don't inadvertently change the original model
        onnx_model = copy.deepcopy(base_model)

        # determine the reduction type
        if reduction != 'mean' and reduction != 'sum':
            raise RuntimeError('not supported reduction')

        graph_nodes = onnx_model.graph.node
        graph_inputs = onnx_model.graph.input

        # create a new graph input. this is the target input needed to compare the
        # graph output against to calculate loss.
        target_input = copy.deepcopy(onnx_model.graph.output[0])
        target_input.name = target
        graph_inputs.append(target_input)

        # create a new graph output for loss
        graph_outputs = [helper.make_tensor_value_info('loss', TensorProto.FLOAT, [1, 1])]

        graph_initializers = onnx_model.graph.initializer

        # loss equation
        # loss = reduce((output-target)^2)

        # create the sub node
        sub_node_input_names = [output, target]
        sub_node_output_names = ['loss_sub_output']
        sub_node =  helper.make_node("Sub",
                                    sub_node_input_names,
                                    sub_node_output_names,
                                    name=f"MSELossSub")
        graph_nodes.append(sub_node)

        # create the square node
        pow_node_input_names = sub_node_output_names
        pow_node_input_names.append('0_pow_exponent')
        pow_node_output_names = ['loss_pow_output']
        pow_node = helper.make_node("Pow",
                                    pow_node_input_names,
                                    pow_node_output_names,
                                    name=f"MSELossPow")
        graph_nodes.append(pow_node)
        graph_initializers.append(helper.make_tensor('0_pow_exponent', TensorProto.FLOAT, [1], [2.0]))

        # create the reduce node
        reduce_node_input_names = pow_node_output_names
        reduce_node_output_names = ['loss']
        reduce_node = helper.make_node("ReduceMean" if reduction == 'mean' else "ReduceSum",
                                    reduce_node_input_names,
                                    reduce_node_output_names,
                                    name=f"MSELossReduce")
        graph_nodes.append(reduce_node)

        # generate the graph and model with above inputs, outputs, initializers and nodes
        graph = helper.make_graph(graph_nodes, 'GraphWithLoss', graph_inputs, graph_outputs, graph_initializers)
        model = helper.make_model(graph, producer_name='onnxflow',
                                  opset_imports=[helper.make_opsetid('com.microsoft', 1)]+ list(base_model.opset_import))

        return model

class CrossEntropyLoss(Graph):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def build(self, base_model, output, target='target', weights=None, reduction='mean', ignore_index=None, get_log_prob=False):

        # Ideally
        # model = onnx_model.make_functional()
        # loss = onnx.SoftmaxCrossEntropyLoss(output, target, weights, reduction)
        # return loss

        # deepcopy the base model so we don't inadvertently change the original model
        onnx_model = copy.deepcopy(base_model)

        # determine the reduction type
        if reduction != 'mean' and reduction != 'sum':
            raise RuntimeError('not supported reduction')

        graph_nodes = onnx_model.graph.node
        graph_inputs = onnx_model.graph.input

        # create a new graph input. this is the target input needed to compare the
        # graph output against to calculate loss.
        target_input = copy.deepcopy(onnx_model.graph.output[0])
        target_input.name = target
        target_input.type.tensor_type.elem_type = TensorProto.INT32
        graph_inputs.append(target_input)

        # create a new graph output for loss
        graph_outputs = [helper.make_tensor_value_info('loss', TensorProto.FLOAT, [])]
        graph_initializers = onnx_model.graph.initializer

        # create the loss node
        loss_node_input_name = [output, target]
        if weights:
            loss_node_input_name.append('weights')
        loss_node_output_name = ['loss', 'log_prob']
        loss_node =  helper.make_node("SoftmaxCrossEntropyLoss",
                                    loss_node_input_name,
                                    loss_node_output_name,
                                    reduction=reduction,
                                    ignore_index=ignore_index,
                                    name=f"SoftmaxCrossEntropyLoss")
        graph_nodes.append(loss_node)

        # generate the graph and model with above inputs, outputs, initializers and nodes
        # TODO: user model generated by opset 11 does not have SoftmaxCrossEntropyLoss.
        #       we need to probably enfore opset versions.
        graph = helper.make_graph(graph_nodes, 'GraphWithLoss', graph_inputs, graph_outputs, graph_initializers)
        model = helper.make_model(graph, producer_name='onnxflow',
                                  opset_imports=[onnx.helper.make_opsetid("", 12)])

        return model

# TODO: BCEWithLogitsLoss

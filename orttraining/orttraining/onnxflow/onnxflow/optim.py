from .graph import Graph
import onnx
import onnx
from onnx import helper
from onnx import TensorProto, OperatorSetIdProto
import copy


class AdamW(Graph):
    def __init__(self, bias_correction=True, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.):
        super(AdamW, self).__init__()
        self.bias_correction = bias_correction
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # TODO: fix this to move outside of optimizer node in ORT backend
        self.max_norm_clip = 1

    def build(self, base_model):
        # Ideally
        # model = onnx_model.make_functional()
        # loss_unreduced = onnx.Pow(onnx.Sub(model(), target), 2)
        # if reduction == 'mean':
        #     loss = onnx.ReduceMean(loss_unreduced)
        # elif reduction == 'sum':
        #     loss = onnx.ReduceSum(loss_unreduced)
        # return loss

        learning_rate_name = 'learning_rate'
        step_name = 'step'

        graph_nodes = []
        graph_inputs = [
            helper.make_tensor_value_info(learning_rate_name, TensorProto.FLOAT , [1]),
            helper.make_tensor_value_info(step_name, TensorProto.INT64, [1])]
        graph_outputs = []

        for idx, graph_output in enumerate(base_model.graph.output):
            if not graph_output.name.endswith('_grad'):
                continue

            weight_name = graph_output.name[:-5]
            grad_name = graph_output.name
            first_order_moment_name = weight_name + '.exp_avg'
            second_order_moment_name = weight_name + '.exp_avg_sq'
            mixed_precision_name = weight_name + '.mixed_precision'
            loss_scaler_name = weight_name + '.loss_scaler'
            gradient_norm_name = weight_name + '.global_gradient_norm'
            should_update_name = weight_name + '.should_update'
            # prepare node (and graph) inputs and outputs
            node_input_names = [learning_rate_name, # learning rate
                                step_name, # training step (used for beta correction)
                                weight_name, # weight to be updated
                                grad_name, # gradient of the weight to be used for update
                                first_order_moment_name, # first order moment for this weight
                                second_order_moment_name, # second order moment for this weight
                                mixed_precision_name, # mixed precision weight representation (required if computation to be done in mp)
                                loss_scaler_name, # used for gradient scaling
                                gradient_norm_name, # used for gradient scaling
                                should_update_name] # whether or not to skip updating the weights

            weight_tensor_value_info = copy.deepcopy(graph_output)
            weight_tensor_value_info.name = weight_name
            first_order_moment_tensor_value_info = copy.deepcopy(graph_output)
            first_order_moment_tensor_value_info.name = first_order_moment_name
            second_order_moment_tensor_value_info = copy.deepcopy(graph_output)
            second_order_moment_tensor_value_info.name = second_order_moment_name
            node_inputs = [
                weight_tensor_value_info,
                copy.deepcopy(graph_output),
                first_order_moment_tensor_value_info,
                second_order_moment_tensor_value_info,
                helper.make_tensor_value_info(mixed_precision_name, TensorProto.FLOAT16 , [0]),
                helper.make_tensor_value_info(loss_scaler_name, TensorProto.FLOAT, []),
                helper.make_tensor_value_info(gradient_norm_name, TensorProto.FLOAT, []),
                helper.make_tensor_value_info(should_update_name, TensorProto.BOOL, [1]),
            ]
            graph_inputs.extend(node_inputs)

            step_output_name = f'{weight_name}.{step_name}.out'
            first_order_moment_output_name = f'{first_order_moment_name}.out'
            second_order_moment_output_name = f'{second_order_moment_name}.out'
            weight_output_name = f'{weight_name}.out'
            grad_output_name = f'{grad_name}.out'
            mixed_precision_output_name = f'{mixed_precision_name}.out'

            first_order_moment_output_tensor_value_info = copy.deepcopy(graph_output)
            first_order_moment_output_tensor_value_info.name = first_order_moment_output_name
            second_order_moment_output_tensor_value_info = copy.deepcopy(graph_output)
            second_order_moment_output_tensor_value_info.name = second_order_moment_output_name
            weight_output_tensor_value_info = copy.deepcopy(graph_output)
            weight_output_tensor_value_info.name = weight_output_name
            grad_output_tensor_value_info = copy.deepcopy(graph_output)
            grad_output_tensor_value_info.name = grad_output_name


            node_output_names = [step_output_name, # step out
                                 first_order_moment_output_name, # first order moment output
                                 second_order_moment_output_name, # second order moment output
                                 weight_output_name, # updated weights
                                 grad_output_name, # gradients output
                                 mixed_precision_output_name] # updated mixed precision weights

            node_outputs = [
                helper.make_tensor_value_info(step_output_name, TensorProto.INT64, [1]),
                first_order_moment_output_tensor_value_info,
                second_order_moment_output_tensor_value_info,
                weight_output_tensor_value_info,
                grad_output_tensor_value_info,
                helper.make_tensor_value_info(mixed_precision_output_name, TensorProto.FLOAT16, [0])
            ]
            graph_outputs.extend(node_outputs)

            # node attributes
            node_attributes = {
                'alpha': self.betas[0], # beta1
                'beta': self.betas[1], # beta2
                'lambda': self.weight_decay, # weight decay
                'epsilon': self.eps, # epsilon
                'do_bias_correction': 1 if self.bias_correction else 0, # bias_correction
                'weight_decay_mode': 1, # weight decay mode 1 implies transformers adamw 0 implies pytorch adamw
                'max_norm_clip': self.max_norm_clip # used for gradient scaling
            }

            # gradient scaling equation:
            # if global_gradient_norm > loss_scaler*max_norm_clip: global_gradient_norm / max_norm_clip
            # else: loss_scaler*max_norm_clip

            # make the node
            optimizer_node = helper.make_node("AdamOptimizer",
                                              node_input_names,
                                              node_output_names,
                                              name=f"AdamOptimizer{idx}",
                                              domain='com.microsoft',
                                              **node_attributes)

            graph_nodes.append(optimizer_node)

        # make the graph and the model
        graph = helper.make_graph(graph_nodes, 'AdamOptimizerGraph', graph_inputs, graph_outputs)
        model = helper.make_model(graph, producer_name='onnxflow',
                                  opset_imports=[helper.make_opsetid('com.microsoft', 1)])
        return model

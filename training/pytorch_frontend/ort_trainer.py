import io
import numpy as np
import onnx
from onnx import numpy_helper
from onnx import helper
import torch
import torch.nn
import torch.onnx
import onnxruntime as ort
import pdb

class model_loss_cls(torch.nn.Module):
    def __init__(self, model, loss_fn):
        super(model_loss_cls, self).__init__()
        self.model_ = model
        self.loss_fn_ = loss_fn

    def forward(self, input, target):
        preds = self.model_(input)
        return self.loss_fn_(preds, target)

def FuseSofmaxNLLToSoftmaxCE(onnx_model):
    # 
    nll_loss_node = None
    nll_loss_node_index = 0
    for node in onnx_model.graph.node:
        if node.op_type == "nll_loss":
            nll_loss_node = node
            break
        nll_loss_node_index = nll_loss_node_index+1

    if nll_loss_node is None:
        return onnx_model
    
    softmax_node = None
    softmax_node_index = 0
    target_input_name = None
    for node in onnx_model.graph.node:
        if node.op_type == "LogSoftmax":
            if node.output[0] == nll_loss_node.input[0]:
                softmax_node = node
                target_input_name = nll_loss_node.input[1]
                break
            elif node.output[0] == nll_loss_node.input[1]:
                softmax_node = node
                target_input_name = nll_loss_node.input[0]
                break
        else:
            if softmax_node is not None:
                break
        softmax_node_index = softmax_node_index + 1
                
    if softmax_node is None:
        return onnx_model

    if nll_loss_node_index < softmax_node_index:
        del onnx_model.graph.node[softmax_node_index]
        del onnx_model.graph.node[nll_loss_node_index]
    else:
        del onnx_model.graph.node[nll_loss_node_index]
        del onnx_model.graph.node[softmax_node_index]

    prediction = onnx.helper.make_tensor_value_info(
        softmax_node.input[0],
        1,      # float32
        ['batch', 10])
    onnx_model.graph.output.extend([prediction])

    # add addition probability output to the softmaxCE node
    probability_output = onnx.helper.make_tensor_value_info(
        "probability", 
        1,      # float32
        ['batch', 10])
    #onnx_model.graph.output.extend([probability_output])

    node = onnx_model.graph.node.add()
    node.CopyFrom(onnx.helper.make_node("SparseSoftmaxCrossEntropy", [softmax_node.input[0], target_input_name], 
        [nll_loss_node.output[0], probability_output.name], "nll_loss_node"))

    return onnx_model
    
def delete_input_with_name(input, name):
    index = 0
    for i in input:
        if i.name == name:
            del input[index]
            break
        index = index + 1

# reference:
# https://docs.scipy.org/doc/numpy-1.13.0/user/basics.types.html
# https://pytorch.org/docs/stable/tensors.html
# also must map to types accepted by: 
# MLDataType NumpyTypeToOnnxRuntimeType(int numpy_type)
def dtype_torch_to_numpy(torch_dtype):
    if torch_dtype == torch.float64 or torch_dtype == torch.double:
        return np.float64
    elif torch_dtype == torch.float32 or torch_dtype == torch.float:
        return np.float32
    elif torch_dtype == torch.float16 or torch_dtype == torch.half:
        return np.float16
    elif torch_dtype == torch.int64 or torch_dtype == torch.long: 
        return np.longlong
    elif torch_dtype == torch.int32 or torch_dtype == torch.int: 
        return np.int32
    elif torch_dtype == torch.int16 or torch_dtype == torch.short: 
        return np.int16

class ORTTrainer():
    def __init__(self, model, loss_fn, optimizer_constructor_lambda):
        super(ORTTrainer, self).__init__()
        self.model_ = model
        self.loss_fn_ = loss_fn
        self.optimizer_constructor_lambda_ = optimizer_constructor_lambda
        self.input_name = 'inp'
        self.label_name = 'lbl'

        self.optimizer_ = None
        self.session = None

    def compile_torch(self):
        self.optimizer_ = self.optimizer_constructor_lambda_(self.model_.parameters())

    def GenerateSampleDataForJit(self, batch_size, feature_shape, input_type, number_classes, label_type):
        input = torch.randn(batch_size, feature_shape, dtype=input_type).cuda() 
        target = torch.randint(0, number_classes, (batch_size,), dtype=label_type).cuda()
        return input, target

    def compile_ort(self, batch_size, feature_shape, input_dtype, number_classes, label_type):
        input, target = self.GenerateSampleDataForJit(batch_size, feature_shape, input_dtype, number_classes, label_type)
        # combine model and loss and create the combined ONNX model
        model_loss = model_loss_cls(self.model_, self.loss_fn_)
        self.dynamic_axes = {self.input_name:{0:'batch'}, self.label_name:{0:'batch'}}

        f = io.BytesIO()

        torch.onnx._export(model_loss, (input, target), f,
            input_names = [self.input_name, self.label_name],
            opset_version=12,
            dynamic_axes = self.dynamic_axes,
            example_outputs=torch.zeros((1,), dtype=torch.float))
        
        model = onnx.load_model_from_string(f.getvalue())

        self.output_name = model.graph.output[0].name
        # onnx.save_model(model, "ORTTrainer2_model_loss.onnx")

        model = FuseSofmaxNLLToSoftmaxCE(model)

        self.prediction_name = model.graph.output[1].name

        self.ort_parameters = ort.TrainingParameters()
        self.ort_parameters.loss_output_name = self.output_name
        self.ort_parameters.enable_mix_precision = False
        
        self.torch_params = {}
        self.output_types = {}
        for initializer in model.graph.initializer:
            torch_tensor = torch.nn.Parameter(torch.from_numpy(numpy_helper.to_array(initializer)).cuda())
            delete_input_with_name(model.graph.input, initializer.name)
            model.graph.input.extend([helper.make_tensor_value_info(initializer.name, initializer.data_type, initializer.dims)])
            self.torch_params[initializer.name] = torch_tensor

        del model.graph.initializer[:]

        for output in model.graph.output:
            self.output_types[output.name] = output.type.tensor_type

        self.ort_parameters.weights_to_train = set(self.torch_params.keys())
        
        self.session = ort.TrainingSession(model.SerializeToString(), self.ort_parameters)


        self.train_io_binding = self.session.io_binding()
        self.eval_io_binding = self.session.io_binding()
        for param in self.torch_params.keys():
            torch_tensor = self.torch_params[param]

            self.train_io_binding.bind_input(param, torch_tensor.device.type, torch_tensor.device.index, 
                dtype_torch_to_numpy(input_dtype), list(torch_tensor.size()), torch_tensor.data_ptr())
            self.eval_io_binding.bind_input(param, torch_tensor.device.type, torch_tensor.device.index, 
                dtype_torch_to_numpy(input_dtype), list(torch_tensor.size()), torch_tensor.data_ptr())
            torch_tensor.grad = torch.zeros(torch_tensor.size(), dtype=torch.float32).cuda()
            #hardcode grad name
            self.train_io_binding.bind_output(param + "_grad",
                                       torch_tensor.grad.device.type,
                                       torch_tensor.grad.device.index,
                                       np.float32,
                                       list(torch_tensor.grad.size()),
                                       torch_tensor.grad.data_ptr())

        self.optimizer_ = self.optimizer_constructor_lambda_(list(self.torch_params.values()))

    def compile(self, batch_size, feature_shape, input_dtype, number_classes, label_type, use_ort_backend = True):
        if use_ort_backend:
            self.compile_ort(batch_size, feature_shape, input_dtype, number_classes, label_type)
        else:
            self.compile_torch()

    def _resolve_dims(self, input_shapes):
        result = {}
        for name in input_shapes:
            if name in self.dynamic_axes.keys():
                shape = input_shapes[name]
                dynamic_dims = self.dynamic_axes[name]
                for index in dynamic_dims.keys():
                    result[dynamic_dims[index]] = shape[index]
        return result

    def train_step(self, input, label, fetches = None):
        if self.optimizer_ is None:
            self.compile_torch()
            
        self.optimizer_.zero_grad()

        if self.session is None:
            self.model_.train()
            output = self.model_(input)
            loss = self.loss_fn_(output, label)
            loss.backward()
            self.optimizer_.step()
            return loss
        else:
            if fetches is None:
                fetches = [self.ort_parameters.loss_output_name]

            self.train_io_binding.bind_input(self.input_name, input.device.type, input.device.index, dtype_torch_to_numpy(input.dtype),
                                        list(input.size()), input.data_ptr())

            self.train_io_binding.bind_input(self.label_name, label.device.type, label.device.index, dtype_torch_to_numpy(label.dtype),
                                        list(label.size()), label.data_ptr())
            dynamic_dims = self._resolve_dims({self.input_name : list(input.size()), self.label_name : list(label.size())})

            # only loss output is fetched for now. 
            torch_outputs = {}
            for fetch_name in fetches:
                dims = self.output_types[fetch_name].shape.dim
                shape = [dynamic_dims[_.dim_param] if _.dim_param else _.dim_value for _ in dims]
                torch_tensor = torch.zeros(shape, device='cuda', dtype=torch.float32)
                self.train_io_binding.bind_output(fetch_name,torch_tensor.device.type, torch_tensor.device.index, np.float32, list(torch_tensor.size()), torch_tensor.data_ptr())
                torch_outputs[fetch_name] = torch_tensor

            self.session.run_with_iobinding(self.train_io_binding)
            self.optimizer_.step()
            
            if len(fetches) == 1:
                # TODO: most time it returns loss tensor for caller to report training progress.
                return torch_outputs[fetches[0]]
            else:
                return torch_outputs

    def eval(self, input, fetches = None):
        if self.session is None:
            self.model_.eval()
            return self.model_(input)
        else:
            if fetches is None:
                fetches = [self.prediction_name]

            self.eval_io_binding.bind_input(self.input_name, input.device.type, input.device.index,
                                             dtype_torch_to_numpy(input.dtype),
                                             list(input.size()), input.data_ptr())

            dynamic_dims = self._resolve_dims(
                {self.input_name: list(input.size())})

            torch_outputs = {}
            for fetch_name in fetches:
                dims = self.output_types[fetch_name].shape.dim
                shape = [dynamic_dims[_.dim_param] if _.dim_param else _.dim_value for _ in dims]
                torch_tensor = torch.zeros(shape, device='cuda', dtype=torch.float32)
                self.eval_io_binding.bind_output(fetch_name,torch_tensor.device.type, torch_tensor.device.index, np.float32, list(torch_tensor.size()), torch_tensor.data_ptr())
                torch_outputs[fetch_name] = torch_tensor

            run_options = ort.RunOptions()
            run_options.only_execute_path_to_fetches = True
            self.session.run_with_iobinding(self.eval_io_binding, run_options)
            if len(fetches) == 1:
                # TODO: most time it returns loss tensor for caller to report training progress.
                return torch_outputs[fetches[0]]
            else:
                return torch_outputs
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

class IODescription():
    def __init__(self, name, shape, dtype, num_classes=None):
        self.name_ = name
        self.shape_ = shape
        self.dtype_ = dtype
        self.num_classes_ = num_classes


class ModelDescription():
    def __init__(self, inputs, outputs):
        self.inputs_ = inputs
        self.outputs_ = outputs


def resolve_symbolic_dimensions(inputs, input_descs, output_descs):
    import copy
    output_descs_copy = copy.deepcopy(output_descs)
    resolved_dims = {}
    for input, input_desc in zip(inputs, input_descs):
        for i, axis in enumerate(input_desc.shape_):
            if isinstance(axis, str):
                resolved_dims[axis] = input.size()[i]
    
    for output_desc in output_descs_copy:
        for i, axis in enumerate(output_desc.shape_):
            if isinstance(axis, str):
                output_desc.shape_[i] = resolved_dims[axis]

    if any(isinstance(axis, str) for axis in output_desc.shape_ for output_desc in output_descs):
        raise RuntimeError("Cannot run model with unknown output dimensions")

    return output_descs_copy


def generate_sample(desc, device=None):
    # symbolic dimensions are described with strings. set symbolic dimensions to be 1
    size = [s if isinstance(s, (int)) else 1 for s in desc.shape_]
    if desc.num_classes_:
        return torch.randint(0, desc.num_classes_, size, dtype=desc.dtype_, device=device)
    else: 
        return torch.randn(size, dtype=desc.dtype_, device=device)


def ort_training_session_run_helper(session, iobinding, inputs, input_descs, output_descs, device, run_options=None):
    for input, input_desc in zip(inputs, input_descs):
        device_index = input.device.index if input.device.index else 0
        iobinding.bind_input(input_desc.name_, input.device.type, device_index, dtype_torch_to_numpy(input.dtype),
                                list(input.size()), input.data_ptr())

    output_descs_resolved = resolve_symbolic_dimensions(inputs, input_descs, output_descs)
    torch_outputs = {}
    for output_desc in output_descs_resolved:
        torch_tensor = torch.zeros(output_desc.shape_, device=device, dtype=output_desc.dtype_)
        device_index = device.index if device.index else 0
        iobinding.bind_output(output_desc.name_, torch_tensor.device.type, device_index, 
                              dtype_torch_to_numpy(torch_tensor.dtype),  
                              list(torch_tensor.size()), torch_tensor.data_ptr())
        torch_outputs[output_desc.name_] = torch_tensor

    session.run_with_iobinding(iobinding, run_options)
    return torch_outputs


class model_loss_cls(torch.nn.Module):
    def __init__(self, model, loss_fn):
        super(model_loss_cls, self).__init__()
        self.model_ = model
        self.loss_fn_ = loss_fn

    def forward(self, *inputs):
        # here we assume input can be unpacked into input and label
        input, label = inputs[:-1], inputs[-1]
        preds = self.model_(*input)
        return self.loss_fn_(preds, label), preds


def FuseSofmaxNLLToSoftmaxCE(onnx_model):
    nll_count = 0
    while True:
        nll_count = nll_count + 1
        nll_loss_node = None
        nll_loss_node_index = 0
        for nll_loss_node_index, node in enumerate(onnx_model.graph.node):
            if node.op_type == "nll_loss":
                nll_loss_node = node
                break

        if nll_loss_node is None:
            break
        
        softmax_node = None
        softmax_node_index = 0
        label_input_name = None
        weight_input_name = None
        for softmax_node_index, node in enumerate(onnx_model.graph.node):
            if node.op_type == "LogSoftmax":
                # has to be connected to nll_loss
                if len(nll_loss_node.input) > 2:
                    weight_input_name = nll_loss_node.input[2]
                if node.output[0] == nll_loss_node.input[0]:
                    softmax_node = node
                    label_input_name = nll_loss_node.input[1]
                    break
                elif node.output[0] == nll_loss_node.input[1]:
                    softmax_node = node
                    label_input_name = nll_loss_node.input[0]
                    break
            else:
                if softmax_node is not None:
                    break
                    
        if softmax_node is None:
            break

        # delete nll_loss and LogSoftmax nodes in order
        if nll_loss_node_index < softmax_node_index:
            del onnx_model.graph.node[softmax_node_index]
            del onnx_model.graph.node[nll_loss_node_index]
        else:
            del onnx_model.graph.node[nll_loss_node_index]
            del onnx_model.graph.node[softmax_node_index]

        probability_output_name = softmax_node.output[0]
        node = onnx_model.graph.node.add()
        inputs = [softmax_node.input[0], label_input_name, weight_input_name] if weight_input_name else [softmax_node.input[0], label_input_name]
        node.CopyFrom(onnx.helper.make_node("SparseSoftmaxCrossEntropy", inputs, 
            [nll_loss_node.output[0], probability_output_name], "nll_loss_node_" + str(nll_count)))

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


def convert_model_loss_fn_to_onnx(model, loss_fn, model_desc, device):
    # example: {input0:{0:'batch'}, input1:{0:'batch'}}
    dynamic_axes = {}
    for input in model_desc.inputs_:
        symbolic_axis = {}
        for i, axis in enumerate(input.shape_):
            if isinstance(axis, str):
                symbolic_axis[i] = axis
        if len(symbolic_axis):
            dynamic_axes[input.name_] = symbolic_axis

    for output in model_desc.outputs_:
        symbolic_axis = {}
        for i, axis in enumerate(output.shape_):
            if isinstance(axis, str):
                symbolic_axis[i] = axis
        if len(symbolic_axis):
            dynamic_axes[output.name_] = symbolic_axis

    input_names = [input.name_ for input in model_desc.inputs_]
    output_names = [output.name_ for output in model_desc.outputs_] 
    
    sample_inputs = []
    for input_desc in model_desc.inputs_:
        input_sample = generate_sample(input_desc, device)
        sample_inputs.append(input_sample)
    
    sample_outputs = []
    for output_desc in model_desc.outputs_:
        output_sample = generate_sample(output_desc, device)
        sample_outputs.append(output_sample)
            
    f = io.BytesIO()

    if loss_fn:
        model = model_loss_cls(model, loss_fn)
        
    torch.onnx._export(model, tuple(sample_inputs), f,
                       input_names=input_names, 
                       output_names=output_names,
                       opset_version=10,
                       dynamic_axes=dynamic_axes,
                       training=True,
                       _retain_param_name=True,
                       example_outputs=tuple(sample_outputs))
    
    model = onnx.load_model_from_string(f.getvalue())

    model = FuseSofmaxNLLToSoftmaxCE(model)
    return model


def create_ort_training_session_bind_parameters(model, device, enable_grad_accumulation=False):
    output_name = model.graph.output[0].name
    ort_parameters = ort.TrainingParameters()
    ort_parameters.loss_output_name = output_name
    ort_parameters.enable_mix_precision = False
    ort_parameters.enable_grad_accumulation = enable_grad_accumulation

    torch_params = {}
    for initializer in model.graph.initializer:
        torch_tensor = torch.nn.Parameter(torch.as_tensor(numpy_helper.to_array(initializer), device=device))
        delete_input_with_name(model.graph.input, initializer.name)
        model.graph.input.extend(
            [helper.make_tensor_value_info(initializer.name, initializer.data_type, initializer.dims)])
        torch_params[initializer.name] = torch_tensor

    del model.graph.initializer[:]

    output_types = {}
    for output in model.graph.output:
        output_types[output.name] = output.type.tensor_type

    ort_parameters.weights_to_train = set(torch_params.keys())

    if device.type == 'cuda' and hasattr(device, "index") and device.index is not None:
        from onnxruntime.capi._pybind_state import set_cuda_device_id 
        set_cuda_device_id(device.index) 
    session = ort.TrainingSession(model.SerializeToString(), ort_parameters)

    train_io_binding = session.io_binding()
    eval_io_binding = session.io_binding()
    grad_buffers = {} if enable_grad_accumulation else None
    for param in torch_params.keys():
        torch_tensor = torch_params[param]

        device_index = torch_tensor.device.index if torch_tensor.device.index else 0
        train_io_binding.bind_input(param, torch_tensor.device.type, device_index,
                                    dtype_torch_to_numpy(torch_params[param].dtype), list(torch_tensor.size()),
                                    torch_tensor.data_ptr())
        eval_io_binding.bind_input(param, torch_tensor.device.type, device_index,
                                   dtype_torch_to_numpy(torch_params[param].dtype), list(torch_tensor.size()),
                                   torch_tensor.data_ptr())
        # hardcode grad name
        grad_buffer_name = (param + "_grad") if enable_grad_accumulation is False else (param + "_grad_accumulate_buffer")
        if torch_tensor.grad is None:
            torch_tensor.grad = torch.zeros(torch_tensor.size(), dtype=torch.float32, device=device)
        if enable_grad_accumulation:
            train_io_binding.bind_input(grad_buffer_name,
                                         torch_tensor.grad.device.type,
                                         device_index,
                                         dtype_torch_to_numpy(torch_tensor.grad.dtype),
                                         list(torch_tensor.grad.size()),
                                         torch_tensor.grad.data_ptr())
        else:
            train_io_binding.bind_output(grad_buffer_name,
                                         torch_tensor.grad.device.type,
                                         device_index,
                                         dtype_torch_to_numpy(torch_tensor.grad.dtype),
                                         list(torch_tensor.grad.size()),
                                         torch_tensor.grad.data_ptr())
    return session, train_io_binding, eval_io_binding, output_name, ort_parameters, torch_params, output_types


class ORTTrainer():
    def __init__(self, model, loss_fn, model_desc, optimizer_constructor_lambda, \
                 device, use_ort_backend=True, accumulate_steps = 1):
        super(ORTTrainer, self).__init__()
        self.model_ = model
        self.loss_fn_ = loss_fn
        self.model_desc_ = model_desc
        self.optimizer_constructor_lambda_ = optimizer_constructor_lambda

        self.optimizer_ = None
        self.session = None
        self.device_ = device
        self.accumulate_steps = accumulate_steps
        self.current_step = 0

        if use_ort_backend:
            model = convert_model_loss_fn_to_onnx(self.model_, self.loss_fn_, self.model_desc_, device)
            self.session, self.train_io_binding, self.eval_io_binding, self.output_name, self.ort_parameters, self.torch_params, self.output_types = \
                create_ort_training_session_bind_parameters(model, device, True if self.accumulate_steps > 1 else False)
            self.optimizer_ = self.optimizer_constructor_lambda_(list(self.torch_params.values()))
            self.device_ = device
        else:
            self.optimizer_ = self.optimizer_constructor_lambda_(self.model_.parameters())

    def train_step(self, input, label, fetches=None):
        self.current_step += 1
        device_index = input.device.index if input.device.index else 0
        if self.session is None:
            self.model_.train()
            output = self.model_(input)
            loss = self.loss_fn_(output, label)
            loss.backward()
            if self.current_step % self.accumulate_steps == 0:
                self.optimizer_.step()
                self.optimizer_.zero_grad()
            return loss
        else:

            session_run_results = ort_training_session_run_helper(self.session, self.train_io_binding, (input, label), \
                                                                  self.model_desc_.inputs_, self.model_desc_.outputs_, 
                                                                  self.device_)
            if self.current_step % self.accumulate_steps == 0:
                self.optimizer_.step()
                self.optimizer_.zero_grad()
            if not fetches or len(fetches) == 1:
                # TODO: most time it returns loss tensor for caller to report training progress.
                return list(session_run_results.values())[0]
            else:
                return (session_run_results[fetch] for fetch in fetches)

    def eval(self, input, fetches=None):
        if self.session is None:
            self.model_.eval()
            return self.model_(input)
        else:
            if fetches is None:
                fetches = [output.name_ for output in self.model_desc_.outputs_[1:]]

            if not isinstance(input, list):
                input = [input]
            # with model_loss_cls, the last input is label, first output is loss
            # TODO: assert size matches

            run_options = ort.RunOptions()
            run_options.only_execute_path_to_fetches = True

            session_run_results = ort_training_session_run_helper(self.session, self.eval_io_binding, (input), \
                                                                  self.model_desc_.inputs_[:-1], 
                                                                  self.model_desc_.outputs_[1:], self.device_, 
                                                                  run_options)

            if len(fetches) == 1:
                # TODO: most time it returns loss tensor for caller to report training progress.
                return session_run_results[fetches[0]]
            else:
                return session_run_results


class ORTModel():
    def __init__(self, model, loss_fn, model_desc, device, postprocess_model=None):
        super(ORTModel, self).__init__()
        self.model_ = model
        self.loss_fn_ = loss_fn
        self.model_desc_ = model_desc
        self.device_ = device

        model = convert_model_loss_fn_to_onnx(self.model_, self.loss_fn_, self.model_desc_, torch.device('cpu'))
        if postprocess_model:
            postprocess_model(model)
        # onnx.save_model(model, 'bert_model_base_after_postproc.onnx')
        self.session_, self.train_io_binding, self.eval_io_binding, self.output_name, self.ort_parameters, self.torch_params, self.output_types = \
            create_ort_training_session_bind_parameters(model, device, True)

    def parameters(self):
        return list(self.torch_params.values())

    def named_parameters(self):
        return self.torch_params.items()

    def train(self):
        pass

    def eval(self):
        pass

    def state_dict(self):
        return self.torch_params

    def load_state_dict(self, state_dict, strict=False):
        for name, param in self.torch_params.items():
            input_param = state_dict[name]
            param.data.copy_(input_param.data)

    def _train(self, *inputs):
        #confirm does the grad buffer binded or not
        for param in self.torch_params.keys():
            torch_tensor = self.torch_params[param]
            if torch_tensor.grad is None:
                torch_tensor.grad = torch.zeros(torch_tensor.size(), dtype=torch.float32, device=self.device_)
                grad_buffer_name = param + "_grad_accumulate_buffer"
                device_index = torch_tensor.device.index if torch_tensor.device.index else 0
                self.train_io_binding.bind_input(grad_buffer_name,
                                             torch_tensor.grad.device.type,
                                             device_index,
                                             dtype_torch_to_numpy(torch_tensor.grad.dtype),
                                             list(torch_tensor.grad.size()),
                                             torch_tensor.grad.data_ptr())

        return ort_training_session_run_helper(self.session_, self.train_io_binding, inputs,
                                                  self.model_desc_.inputs_, self.model_desc_.outputs_,
                                                  self.device_).values()

    def __call__(self, *inputs):
        return self._train(*inputs)

    def run(self, *inputs):
        return self._train(*inputs)

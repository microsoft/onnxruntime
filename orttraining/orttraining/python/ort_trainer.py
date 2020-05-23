import io
import os
import warnings
import numpy as np
import onnx
from onnx import numpy_helper
from onnx import helper
import torch
import torch.nn
import torch.onnx
import onnxruntime as ort
from distutils.version import LooseVersion
from .checkpointing_utils import list_checkpoint_files, get_checkpoint_name, CombineZeroCheckpoint
import onnxruntime.capi.pt_patch

DEFAULT_OPSET_VERSION = 10

class IODescription():
    def __init__(self, name, shape, dtype=None, num_classes=None):
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
        return torch.randint(0, desc.num_classes_, size, dtype=desc.dtype_).to(device)
    else:
        return torch.randn(size, dtype=desc.dtype_).to(device)

def get_device_index(device):
    if type(device) == str:
        # could be 'cuda:0', 'cuda:1', or 'cpu'. with cpu, set index=0
        device = torch.device(device)
    return 0 if device.index is None else device.index

def input_get_device_index(input):
    if isinstance(input, (list, tuple)):
        device_index = get_device_index(input[0].device)
    else:
        device_index = get_device_index(input.device)

    return device_index

def get_all_gradients_finite_arg_name(session):
    all_fp16_or_fp32_gradients_finite_node_args = [x for x in session._outputs_meta if 'all_gradients_finite' in x.name]
    if len(all_fp16_or_fp32_gradients_finite_node_args) != 1:
        raise RuntimeError("Failed to find a group NodeArg with name that matches 'all_gradients_finite'\
             from the training session.")

    return all_fp16_or_fp32_gradients_finite_node_args[0].name

def get_group_accumulated_gradients_output_node_arg_name(session):
    # TODO: get the constant string via pybind.
    # optimizer_graph_builder BuildGroupNode with fixed string: 'Group_Accumulated_Gradients'
    accumulated_gradients_output_node_args = [x for x in session._outputs_meta if 'Group_Accumulated_Gradients' in x.name]
    if len(accumulated_gradients_output_node_args) != 1:
        raise RuntimeError("Failed to find a group NodeArg with name that matches 'Group_Accumulated_Gradients'\
             from the training session.")

    return accumulated_gradients_output_node_args[0].name

def ort_training_session_run_helper(session, iobinding, inputs, input_descs, output_descs, device, run_options=None):
    for input, input_desc in zip(inputs, input_descs):
        device_index = input_get_device_index(input)
        iobinding.bind_input(input_desc.name_, input.device.type, device_index, dtype_torch_to_numpy(input.dtype),
                             list(input.size()), input.data_ptr())

    output_descs_resolved = resolve_symbolic_dimensions(inputs, input_descs, output_descs)
    torch_outputs = {}
    for output_desc in output_descs_resolved:
        torch_tensor = torch.zeros(output_desc.shape_, device=device,
                                   dtype=output_desc.eval_dtype_ if hasattr(output_desc, 'eval_dtype_')
                                   else output_desc.dtype_)

        iobinding.bind_output(output_desc.name_, torch_tensor.device.type, get_device_index(device),
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
            if node.op_type == "nll_loss" or node.op_type == "NegativeLogLikelihoodLoss":
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
                                            [nll_loss_node.output[0], probability_output_name],
                                            "nll_loss_node_" + str(nll_count)))

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

def wrap_for_input_match(model, input_names):
    import inspect
    sig = inspect.signature(model.forward)
    ordered_list_keys = list(sig.parameters.keys())

    if len(ordered_list_keys) < len(input_names):
        # this is likely the case where input arguments are packed.
        # For example when model_loss_cls is used.
        # TODO: to unpack the input argument.
        return model
    elif len(ordered_list_keys) == len(input_names):
        # in this case, we do not require name match. we will if train_step supports dictionary input
        return model

    if not all(x in ordered_list_keys for x in input_names):
        # model desc has name(s) not matching the model signature. We cannot do anything in this case.
        # better to warning the user.
        return model

    # if input_names match the first ordered_list_keys, there is not need for wrapping
    match = True
    for i, input_name in enumerate(input_names):
        if input_name != ordered_list_keys[i]:
            match = False
            break

    if match:
        return model

    class WrapModel(torch.nn.Module):
        def __init__(self, model, input_names):
            super(WrapModel, self).__init__()
            self.model_ = model
            self.input_names_ = input_names

        def forward(self, *inputs):
            # *inputs is given by torch trace. It is in the order of input_names.
            # model_ takes input in a order (which can be obtained via inspect.signature(model.forward)) different than input_names.
            sig = inspect.signature(self.model_.forward)
            ordered_list_keys = list(sig.parameters.keys())

            input_dict = {}
            for key in sig.parameters.keys():
                if key in self.input_names_:
                    input_dict[key] = inputs[self.input_names_.index(key)]

            return self.model_(**input_dict)

    model = WrapModel(model, input_names)
    return model

def convert_model_loss_fn_to_onnx(model, loss_fn, model_desc, device, inputs, opset_version=DEFAULT_OPSET_VERSION):
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

    if isinstance(inputs, torch.Tensor):
        inputs = [inputs]
    if isinstance(inputs, dict):
        sample_inputs = [inputs[k.name_].to(device=device) for k in model_desc.inputs_]
    elif isinstance(inputs, (list, tuple)):
        sample_inputs = [input.to(device=device) for i, input in enumerate(inputs) if i < len(model_desc.inputs_)]
    else:
        raise RuntimeError("Unexpected input type. Only torch.Tensor, or dict/list/tuple of torch.Tensor is supported.")

    if loss_fn:
        model = model_loss_cls(model, loss_fn)

    # pytorch onnx exporter/trace does not try to match argument names.
    # e.g. for models with optional inputs, it requires all inputs be present.
    # this is a problem because the model graph depends on inputs provided.
    model = wrap_for_input_match(model, input_names)

    model.eval()
    with torch.no_grad():
        sample_outputs = model(*sample_inputs)
    if isinstance(sample_outputs, torch.Tensor):
        sample_outputs = [sample_outputs]
    for sample_output, output_desc in zip(sample_outputs, model_desc.outputs_):
        output_desc.dtype_ = sample_output.dtype
    model.train()

    f = io.BytesIO()

    # Other export options to use(this is for backward compatibility).
    other_export_options = {}

    # This option was added after 1.4 release.
    if LooseVersion(torch.__version__) > LooseVersion('1.4.0'):
        other_export_options['enable_onnx_checker'] = False

    torch.onnx._export(model, tuple(sample_inputs), f,
                       input_names=input_names,
                       output_names=output_names,
                       opset_version=opset_version,
                       dynamic_axes=dynamic_axes,
                       training=True,
                       _retain_param_name=True,
                       example_outputs=tuple(sample_outputs),
                       do_constant_folding=False,
                       **other_export_options)

    onnx_model = onnx.load_model_from_string(f.getvalue())

    # Remove 'model_.' prefix introduced by model wrapper for initializers.
    replace_name_dict = {}
    for n in onnx_model.graph.initializer:
        if n.name.startswith('model_.'):
            replace_name_dict[n.name] = n.name[len('model_.'):]
            n.name = replace_name_dict[n.name]
    for n in onnx_model.graph.node:
        for i, name in enumerate(n.input):
            if name in replace_name_dict:
                n.input[i] = replace_name_dict[name]

    # onnx model initializer may contain non-trainable registered buffers that are not part
    # of pytorch model named parameteres.
    assert set([n for n, t in model.model_.named_parameters()]).issubset(
        set([n.name for n in onnx_model.graph.initializer])), \
        "Initializer names do not match between PyTorch model and ONNX model, " \
        "please report a bug to ONNX Runtime."

    onnx_model = FuseSofmaxNLLToSoftmaxCE(onnx_model)

    return onnx_model

def create_ort_training_session_with_optimizer(model, device, training_optimizer_name, lr_params_feed_name,
                                               map_optimizer_attributes, world_rank=-1, world_size=1,
                                               gradient_accumulation_steps=1, bind_parameters=False,
                                               use_mixed_precision=False, allreduce_post_accumulation=False,
                                               partition_optimizer=False,
                                               enable_grad_norm_clip=True,
                                               frozen_weights=[], opset_version=DEFAULT_OPSET_VERSION):
    output_name = model.graph.output[0].name
    ort_parameters = ort.TrainingParameters()
    ort_parameters.loss_output_name = output_name
    ort_parameters.use_mixed_precision = use_mixed_precision
    ort_parameters.world_rank = world_rank
    ort_parameters.world_size = world_size
    ort_parameters.gradient_accumulation_steps = gradient_accumulation_steps
    ort_parameters.use_mixed_precision = use_mixed_precision
    ort_parameters.allreduce_post_accumulation = allreduce_post_accumulation
    ort_parameters.partition_optimizer = partition_optimizer
    ort_parameters.enable_grad_norm_clip = enable_grad_norm_clip
    ort_parameters.set_gradients_as_graph_outputs = False

    output_types = {}
    for output in model.graph.output:
        output_types[output.name] = output.type.tensor_type

    # pybind does not allow to add directly to ort_parameters.weights_to_train.
    # Have to work around by using a temporary weights_to_train.
    torch_params = {}
    optimizer_attributes_map = {}
    optimizer_int_attributes_map = {}

    unused_frozen_weights = [n for n in frozen_weights if n not in [i.name for i in model.graph.initializer]]
    if unused_frozen_weights:
        raise RuntimeError("{} in frozen_weights not found in model weights.".format(unused_frozen_weights))

    weights_to_train = set()
    for initializer in model.graph.initializer:
        if initializer.name in frozen_weights:
            continue
        weights_to_train.add(initializer.name)
        if map_optimizer_attributes is not None:
            attributes = map_optimizer_attributes(initializer.name)
            optimizer_attributes_map[initializer.name] = {}
            optimizer_int_attributes_map[initializer.name] = {}
            for k, v in attributes.items():
                if isinstance(v, float):
                    optimizer_attributes_map[initializer.name][k] = v
                elif isinstance(v, int):
                    optimizer_int_attributes_map[initializer.name][k] = v
                else:
                    raise ValueError("Optimizer attributes must be either float or int.")
        else:
            optimizer_attributes_map[initializer.name] = {}
            optimizer_int_attributes_map[initializer.name] = {}

    if bind_parameters:
        for initializer in model.graph.initializer:
            torch_tensor = torch.nn.Parameter(torch.as_tensor(numpy_helper.to_array(initializer), device=device))
            delete_input_with_name(model.graph.input, initializer.name)
            model.graph.input.extend(
                [helper.make_tensor_value_info(initializer.name, initializer.data_type, initializer.dims)])
            torch_params[initializer.name] = torch_tensor

        del model.graph.initializer[:]

    ort_parameters.weights_to_train = weights_to_train
    ort_parameters.training_optimizer_name = training_optimizer_name
    ort_parameters.lr_params_feed_name = lr_params_feed_name
    ort_parameters.optimizer_attributes_map = optimizer_attributes_map
    ort_parameters.optimizer_int_attributes_map = optimizer_int_attributes_map

    session = ort.TrainingSession(model.SerializeToString(), ort_parameters)
    train_io_binding = session.io_binding()
    eval_io_binding = session.io_binding()

    if bind_parameters:
        for param in torch_params.keys():
            torch_tensor = torch_params[param]

            train_io_binding.bind_input(param, torch_tensor.device.type, get_device_index(torch_tensor.device),
                                        dtype_torch_to_numpy(torch_params[param].dtype), list(torch_tensor.size()),
                                        torch_tensor.data_ptr())
            eval_io_binding.bind_input(param, torch_tensor.device.type, get_device_index(torch_tensor.device),
                                       dtype_torch_to_numpy(torch_params[param].dtype), list(torch_tensor.size()),
                                       torch_tensor.data_ptr())

    return session, train_io_binding, eval_io_binding, output_name, torch_params, output_types

def save_checkpoint(model, checkpoint_dir, checkpoint_prefix="ORT_checkpoint", checkpoint_state_dict=None):
    if checkpoint_state_dict==None:
        checkpoint_state_dict={'model': model.state_dict()}
    else:
        checkpoint_state_dict.update({'model': model.state_dict()})

    assert os.path.exists(checkpoint_dir), "ERROR: Checkpoint directory doesn't exist: {}".format(checkpoint_dir)

    checkpoint_name = get_checkpoint_name(checkpoint_prefix, model.partition_optimizer_, model.world_rank, model.world_size)
    checkpoint_file = os.path.join(checkpoint_dir, checkpoint_name)

    if os.path.exists(checkpoint_file):
        warnings.warn("{} already exists, overwriting.".format(checkpoint_file))

    torch.save(checkpoint_state_dict, checkpoint_file)

def _load_single_checkpoint(model, checkpoint_dir, checkpoint_prefix, is_partitioned, strict):
    checkpoint_name = get_checkpoint_name(checkpoint_prefix, is_partitioned, model.world_rank, model.world_size)
    checkpoint_file = os.path.join(checkpoint_dir, checkpoint_name)

    if is_partitioned:
        assert_msg = ("Couldn't find checkpoint file {}." +
            "Optimizer partitioning is enabled using ZeRO. Please make sure that the "+
            "checkpoint file exists for rank {} of {}.").format(checkpoint_file,model.world_rank, model.world_size)
    else:
        assert_msg = "Couldn't find checkpoint file {}.".format(checkpoint_file)

    assert os.path.exists(checkpoint_file), assert_msg

    checkpoint_state = torch.load(checkpoint_file, map_location='cpu')

    model.load_state_dict(checkpoint_state['model'], strict=strict)
    del(checkpoint_state['model'])
    return checkpoint_state

def _load_multi_checkpoint(model, checkpoint_dir, checkpoint_prefix, strict):
    checkpoint_files = list_checkpoint_files(checkpoint_dir, checkpoint_prefix)

    ckpt_agg = CombineZeroCheckpoint(checkpoint_files)
    aggregate_state_dict = ckpt_agg.aggregate_checkpoints()

    model.load_state_dict(aggregate_state_dict, strict=strict)

    # aggregate other keys in the state_dict.
    # Values will be overwritten for matching keys among workers
    all_checkpoint_states=dict()
    for checkpoint_file in checkpoint_files:
        checkpoint_state = torch.load(checkpoint_file, map_location='cpu')
        del(checkpoint_state['model'])
        all_checkpoint_states.update(checkpoint_state)
    return all_checkpoint_states

def load_checkpoint(model, checkpoint_dir, checkpoint_prefix="ORT_checkpoint", strict=False):
    checkpoint_files = list_checkpoint_files(checkpoint_dir, checkpoint_prefix)
    is_partitioned = False
    if len(checkpoint_files) > 1:
        warnings.warn(f"Found more than one file with prefix {checkpoint_prefix} in directory {checkpoint_dir}." +
            "Attempting to load ZeRO checkpoint.")
        is_partitioned = True
    if (not model.partition_optimizer_) and is_partitioned:
        return _load_multi_checkpoint(model, checkpoint_dir, checkpoint_prefix, strict)
    else:
        return _load_single_checkpoint(model, checkpoint_dir, checkpoint_prefix, is_partitioned, strict)

class ORTTrainer():

    def __init__(self, model, loss_fn, model_desc, training_optimizer_name, map_optimizer_attributes,
                 learning_rate_description, device, gradient_accumulation_steps=1, postprocess_model=None,
                 world_rank=0, world_size=1, use_mixed_precision=False, allreduce_post_accumulation=False,
                 global_step=0, get_lr_this_step=None, loss_scaler=None, partition_optimizer=False,
                 enable_grad_norm_clip=True, frozen_weights=[], _opset_version=DEFAULT_OPSET_VERSION):
        super(ORTTrainer, self).__init__()
        """
        Initializes ORTTrainer.

        Params:
            model: either:
                - a PyTorch model: if 'loss_fn' is not provided, the 'model's first output must be the loss.
                    if 'loss_fn' is provided, 'model' and 'loss_fn' are combined as described in 'loss_fn:'
                - an ONNX model: the 'model's first output must be the loss.
            loss_fn: a PyTorch loss function. It takes two inputs [prediction, label] and ouput a loss tensor.
                If provided, 'loss_fn' is combined with the PyTorch 'model' to form a Combined PyTorch model.
                Inputs to the Combined PyTorch model are concatination of the 'model's input and 'loss_fn's label input.
                Outputs of the Combined PyTorch model are concatination of 'loss_fn's loss output and 'model's outputs.
            model_desc: model input and output description. it is used to specify input/output shapes, types, and names.
                'model_desc' must be consistent with the training model.
            training_optimizer_name: one of: 'SGDOptimizer', 'AdamOptimizer', 'LambOptimizer'.
            map_optimizer_attributes: for optimizers with weight dependent parameters,
                'map_optimizer_attributes' maps weight name to a set of optimization parameters.
            learning_rate_description: in form of IODescription(Learning_Rate_Name, [1,], torch.float32).
                Because learning_rate is an input to the training model, Learning_Rate_Name shall be set so that
                there is no name conflict within the above model.
            device: cuda device to store tensors
            gradient_accumulation_steps: number of training steps to accumulate gradients before averaging and applying them (default is 1)
            postprocess_model: a callable to postprocess the ONNX model that is converted from PyTorch (default is None)
            world_rank: rank id used for distributed training (default is 0)
            world_size: number of ranks participating in distributed training (default is 1)
            use_mixed_precision: flag to enable mixed precision (aka fp16) (default is False)
            allreduce_post_accumulation: controls whether overlaping gradient computation with allreduce (default is False)
            partition_optimizer: controls whether to partition the optimizer state (default is False)
        """
        self.is_train = True

        self.torch_model_ = None
        self.onnx_model_ = None
        if isinstance(model, torch.nn.Module):
            self.torch_model_ = model
            self.loss_fn_ = loss_fn
        else:
            self.onnx_model_ = model
            if loss_fn is not None:
                warnings.warn("loss_fn is not used when creating ORTTrainer because an ONNX model is provided.")
            # TODO: accept loss_fn as an onnx model. build self.onnx_model_ with model and loss_fn
            self.loss_fn_ = None

        self.model_desc_ = model_desc
        self.input_desc_with_lr = [*self.model_desc_.inputs_, learning_rate_description]

        self.world_rank = world_rank
        self.world_size = world_size
        self.use_mixed_precision = use_mixed_precision

        self.session = None
        self.device_ = device
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # we use self.current_step to count calls to train_step. It is used for gradient accumulation.
        # gradients are being accumulated when self.current_step is not divisible by gradient_accumulation_steps.
        # gradients are updated when self.current_step is divisible by gradient_accumulation_steps.
        self.current_step = 0

        # we use self.global_step_ to count optimizations being performed.
        # it is used to calculate learning rate if self.get_lr_this_step_ is provided.
        self.global_step_ = global_step
        self.post_process_model_fn_ = postprocess_model
        self.get_lr_this_step_ = get_lr_this_step
        self.loss_scaler_ = loss_scaler

        if self.get_lr_this_step_ is not None or self.loss_scaler_ is not None:
            warnings.warn("It is experimental to use learning rate scheduler and loss scaler inside ORTTrainer.")
        self.training_optimizer_name_ = training_optimizer_name
        self.learning_rate_description_ = learning_rate_description
        self.map_optimizer_attributes_ = map_optimizer_attributes
        self.allreduce_post_accumulation_ = allreduce_post_accumulation
        self.partition_optimizer_ = partition_optimizer
        self.enable_grad_norm_clip_ = enable_grad_norm_clip
        self.frozen_weights_ = frozen_weights
        self.opset_version_ = _opset_version
        self.state_dict_ = None

        # use this special string to workaround a corner case that external loss_scale is passed into train_step as kwargs.
        # see prepare_input_and_fetches for more details.
        self.loss_scale_input_name = 'default_loss_scale_input_name'

        self._init_session()

    def _init_session(self):
        if self.onnx_model_ is None:
            return

        self._verify_fully_optimized_model(self.onnx_model_)
        self.session, self.train_io_binding, self.eval_io_binding, self.output_name, _, self.output_types = \
            create_ort_training_session_with_optimizer(
                self.onnx_model_, self.device_,
                self.training_optimizer_name_, self.learning_rate_description_.name_, self.map_optimizer_attributes_,
                self.world_rank, self.world_size,
                self.gradient_accumulation_steps, bind_parameters=False,
                use_mixed_precision=self.use_mixed_precision, allreduce_post_accumulation=self.allreduce_post_accumulation_,
                partition_optimizer=self.partition_optimizer_,
                enable_grad_norm_clip=self.enable_grad_norm_clip_,
                frozen_weights=self.frozen_weights_, opset_version=self.opset_version_)

        self.loss_scale_input_name = self.session.loss_scale_input_name

        if self.use_mixed_precision:
            self.input_desc_with_lr_and_loss_scale = [
                *self.input_desc_with_lr,
                IODescription(self.loss_scale_input_name, [], torch.float32)]

        # ORT backend has modified model output dtype from float32 to float16.
        for o_desc in self.model_desc_.outputs_:
            if self.use_mixed_precision and o_desc.dtype_ == torch.float32 and not self.session.is_output_fp32_node(o_desc.name_):
                o_desc.eval_dtype_ = torch.float16
            else:
                o_desc.eval_dtype_ = o_desc.dtype_

        # gradient accumulation buffers are connected to a single node with a boolean, dimension 1 tensor output.
        # add a matching output to drive gradient accumulation.
        if self.gradient_accumulation_steps > 1:
            self.output_desc_with_group_accumulated_gradients = [
                *self.model_desc_.outputs_,
                IODescription(get_group_accumulated_gradients_output_node_arg_name(self.session), [1], torch.bool)]

        if self.use_mixed_precision:
            # when ready to use accumulated gradient with mixed precision, we need to fetch all_infinite to determine
            # if the gradient is usable.
            self.output_desc_with_all_fp_16_or_fp32_gradients_finite = [
                *self.model_desc_.outputs_,
                IODescription(get_all_gradients_finite_arg_name(self.session), [1], torch.bool)]

        if self.state_dict_:
            self.load_state_dict(self.state_dict_, self.strict_)
        self.state_dict_ = None

    def _init_onnx_model(self, inputs):
        if self.onnx_model_ is not None:
            return

        if self.torch_model_ is not None:
            # NOTE: pt model is moved to cpu to conserve gpu memory.
            self.torch_model_.cpu()
            self.onnx_model_ = convert_model_loss_fn_to_onnx(
                self.torch_model_, self.loss_fn_, self.model_desc_, torch.device('cpu'), inputs, opset_version=self.opset_version_)

            if self.post_process_model_fn_:
                self.post_process_model_fn_(self.onnx_model_)

        self._init_session()

    def train(self):
        self.is_train = True

    def eval(self):
        self.is_train = False

    def _update_onnx_model_initializers(self, state_tensors):
        # replace the initializers with new value
        new_weights = []
        replace_indices = []
        for i, w in enumerate(self.onnx_model_.graph.initializer):
            if w.name in state_tensors:
                new_weights.append(numpy_helper.from_array(state_tensors[w.name], w.name))
                replace_indices.append(i)
        replace_indices.sort(reverse=True)
        for w_i in replace_indices:
            del self.onnx_model_.graph.initializer[w_i]
        self.onnx_model_.graph.initializer.extend(new_weights)

    def state_dict(self):
        if not self.session:
            warnings.warn("ONNXRuntime training session is not initialized yet. "
                          "Please run train_step or eval_step at least once before calling state_dict().")
            return {}

        # extract trained weights
        session_state = self.session.get_state()
        torch_state = {}
        for name in session_state:
            torch_state[name] = torch.from_numpy(session_state[name])

        # extract untrained weights and buffer
        for n in self.onnx_model_.graph.initializer:
            if n.name not in torch_state:
                torch_state[n.name] = torch.from_numpy(numpy_helper.to_array(n))

        return torch_state

    def load_state_dict(self, state_dict, strict=False):
        # Note: It may happen ONNX model has not yet been initialized
        # In this case we cache a reference to desired state and delay the restore until after initialization
        # Unexpected behavior will result if the user changes the reference before initialization
        if not self.session:
            self.state_dict_ = state_dict
            self.strict_ = strict
            return

        # update onnx model from loaded state dict
        cur_initializers_names = [n.name for n in self.onnx_model_.graph.initializer]
        new_initializers = {}

        for name in state_dict:
            if name in cur_initializers_names:
                new_initializers[name] = state_dict[name].numpy()
            elif strict:
                raise RuntimeError("Checkpoint tensor: {} is not present in the model.".format(name))

        self._update_onnx_model_initializers(new_initializers)

        # create new session based on updated onnx model
        self.state_dict_ = None
        self._init_session()

    def save_as_onnx(self, path):
        if not self.session:
            warnings.warn("ONNXRuntime training session is not initialized yet. "
                          "Please run train_step or eval_step at least once before calling save_as_onnx().")
            return
        state_tensors = self.session.get_state()
        self._update_onnx_model_initializers(state_tensors)

        with open(path, "wb") as f:
            f.write(self.onnx_model_.SerializeToString())

    def _prepare_input_and_fetches(self, input_desc_with_, internal_learning_rate, internal_loss_scale, *args, **kwargs):
        fetches = None
        if type(args) == tuple and len(args) == 1 and type(args[0]) == list:
            input = tuple(args[0])
        else:
            input = args

        for input_desc in input_desc_with_:
            if input_desc.name_ in kwargs:
                input = input + (kwargs[input_desc.name_],)
        if internal_learning_rate is not None:
            input = input + (internal_learning_rate,)
        if internal_loss_scale is not None:
            input = input + (internal_loss_scale,)
        elif self.use_mixed_precision:
            # loss_scale input name is needed to call train_step, for example:
            #   kwargs[model.loss_scale_input_name] = loss_scale
            #   outputs = model.train_step(*args, **kwargs)
            # However, when first time train_step is called model.loss_scale_input_name is not set.
            # To workaround this problem, we use the special name 'default_loss_scale_input_name' to indicate
            # the loss_scale.
            if 'default_loss_scale_input_name' in kwargs.keys():
                input = input + (kwargs['default_loss_scale_input_name'],)

        fetches = None
        if 'fetches' in kwargs:
            fetches = kwargs['fetches']

        return input, fetches

    def train_step(self, *args, **kwargs):
        """
        inputs: model inputs, labels, learning rate, and, if in mixed_precision mode, loss_scale.
        outputs: if fetches is not provided, outputs are loss and
            (if in mixed mode and is finishing gradient accumulation) all_finite.
            if fetches is provided, outputs contains these requested with fetches.
        fetches: names of requested outputs
        """

        # inputs to the ONNX model includes inputs to the original PyTorch model
        # plus learning rate and loss_scale if self.use_mixed_precision is True.
        # 1. when there are internal learning_rate and loss_scale (in fp16 cases) generators,
        #   *args and **kwargs together contain ONLY and COMPLETE inputs to the PyTorch model.
        #   In this case, changes to the training script is minimized.
        # 2. without internal learning rate and loss scale (in fp16 cases) generators,
        #   *args and **kwargs passed in from the training script shall contains
        #   inputs to the PyTorch model plus learning_rate and loss_scale.
        #   it optionally contains the fetches.
        # localized arguments (*args) contains inputs to the ONNX model.
        # named arguments can contain both inputs, learning_rate and loss_scale, and the fetches


        learning_rate, loss_scale = None, None
        if self.get_lr_this_step_ is not None:
            # $args, **kwargs contains inputs to the pytorch model
            lr_this_step = self.get_lr_this_step_(self.global_step_)
            learning_rate = torch.tensor([lr_this_step])
        if self.loss_scaler_ is not None and self.use_mixed_precision:
            loss_scale = torch.tensor([self.loss_scaler_.loss_scale_])

        if self.onnx_model_ is None:
            sample_input, _ = self._prepare_input_and_fetches(self.model_desc_.inputs_,
                None, None, *args, **kwargs)
            self._init_onnx_model(sample_input)

        if self.use_mixed_precision:
            input, fetches = self._prepare_input_and_fetches(self.input_desc_with_lr_and_loss_scale,
                learning_rate, loss_scale, *args, **kwargs)
            assert len(self.input_desc_with_lr_and_loss_scale) == len(input)
            input_descs = self.input_desc_with_lr_and_loss_scale
        else:
            input, fetches = self._prepare_input_and_fetches(self.input_desc_with_lr,
                learning_rate, loss_scale, *args, **kwargs)
            assert len(self.input_desc_with_lr) == len(input)
            input_descs = self.input_desc_with_lr

        self.current_step += 1

        # handle gradient accumulation in fully optimized mode
        run_options = None
        has_if_all_finite = False
        if fetches:
            output_desc = [output for fetch in fetches for output in self.model_desc_.outputs_ if output.name_ == fetch]
        elif self.current_step % self.gradient_accumulation_steps != 0:
            run_options = ort.RunOptions()
            run_options.only_execute_path_to_fetches = True
            run_options.training_mode = True
            output_desc = self.output_desc_with_group_accumulated_gradients
        elif self.use_mixed_precision:
            has_if_all_finite = True
            output_desc = self.output_desc_with_all_fp_16_or_fp32_gradients_finite
        else:
            output_desc = self.model_desc_.outputs_

        if not isinstance(input, (list, tuple)):
            input = (input,)

        session_run_results = ort_training_session_run_helper(self.session, self.train_io_binding, input,
                                                              input_descs, output_desc,
                                                              self.device_,
                                                              run_options)

        if has_if_all_finite:
            # After session run with all_fp32_gradients_finite, we need to clear the iobinding's output state.
            # Otherwise next run with only_execute_path_to_fetches will lead to gradient all reduce
            # because all_fp32_gradients_finite is still in the feed.
            self.train_io_binding.clear_binding_outputs()

            all_finite = session_run_results[self.output_desc_with_all_fp_16_or_fp32_gradients_finite[-1].name_]
            if self.loss_scaler_ is not None:
                self.loss_scaler_.update_loss_scale(all_finite)
            if all_finite:
                # optimization has done, increase self.global_step_
                self.global_step_ = self.global_step_ + 1
        elif self.current_step % self.gradient_accumulation_steps == 0:
            # optimization has done, increase self.global_step_
            self.global_step_ = self.global_step_ + 1

        if fetches is not None:
            results = [session_run_results[fetch] for fetch in fetches]
        elif has_if_all_finite and self.loss_scaler_ is None:
            # return descripted outputs plus the all_finite flag so that the training script can handle loss scaling.
            results = [session_run_results[output_desc.name_] for output_desc in self.output_desc_with_all_fp_16_or_fp32_gradients_finite]
        else:
            results = [session_run_results[output_desc.name_] for output_desc in self.model_desc_.outputs_]

        return results[0] if len(results) == 1 else results

    def __call__(self, *args, **kwargs):
        if self.is_train:
            return self.train_step(*args, **kwargs)
        else:
            return self.eval_step(*args, **kwargs)

    def eval_step(self, *args, **kwargs):
        """
        inputs: model inputs and/or labels.
        outputs: if 'fetches' is not provided, outputs are loss and
            (if in mixed mode and is finishing gradient accumulation) all_finite.
            if fetches is provided, outputs contains these requested with fetches.
        fetches: names of requested outputs
        """

        # with model_loss_cls, the last input is label, first output is loss
        input, fetches = self._prepare_input_and_fetches(self.model_desc_.inputs_,
                                                         None, None, *args, **kwargs)

        if self.onnx_model_ is None:
            if self.torch_model_ is not None:
                self._init_onnx_model(input)
            else:
                raise RuntimeError("Model is unintialized. Please ensure a valid ONNX model or PyTorch model is provided to this Trainer.")

        input_desc = self.model_desc_.inputs_[0:len(input)]
        if fetches is None:
            output_desc = self.model_desc_.outputs_
        else:
            output_desc = [output for fetch in fetches for output in self.model_desc_.outputs_ if output.name_ == fetch]

        if not isinstance(input, (list, tuple)):
            input = (input,)

        run_options = ort.RunOptions()
        run_options.only_execute_path_to_fetches = True
        run_options.training_mode = False

        session_run_results = ort_training_session_run_helper(self.session, self.eval_io_binding, input,
                                                              input_desc,
                                                              output_desc,
                                                              self.device_,
                                                              run_options)

        if len(session_run_results) == 1:
            return session_run_results[list(session_run_results.keys())[0]]
        else:
            return [session_run_results[output_desc.name_] for output_desc in output_desc]

    def _verify_fully_optimized_model(self, model):
        assert(len(model.graph.output) > 0)
        # model's first output must be the loss tensor
        if model.graph.output[0].type.tensor_type.elem_type != onnx.TensorProto().FLOAT and\
                model.graph.output[0].type.tensor_type.elem_type != onnx.TensorProto().FLOAT16 and\
                model.graph.output[0].type.tensor_type.elem_type != onnx.TensorProto().DOUBLE and\
                model.graph.output[0].type.tensor_type.elem_type != onnx.TensorProto().COMPLEX64 and\
                model.graph.output[0].type.tensor_type.elem_type != onnx.TensorProto().COMPLEX128 and\
                model.graph.output[0].type.tensor_type.elem_type != onnx.TensorProto().BFLOAT16:
            raise RuntimeError("the first output of a model to run with fully optimized ORT backend must be float types.")
        if len(model.graph.output[0].type.tensor_type.shape.dim) != 0:
            raise RuntimeError("the first output of a model to run with fully optimized ORT backend must be a scaler.")

class LossScaler():
    def __init__(self, loss_scale_input_name, is_dynamic_scale,
                 loss_scale=float(1 << 16),
                 up_scale_window=2000,
                 min_loss_scale=1.0, max_loss_scale=float(1 << 24)):
        super(LossScaler, self).__init__()
        self.loss_scale_input_name_ = loss_scale_input_name
        self.is_dynamic_scale_ = is_dynamic_scale
        self.initial_loss_scale_ = loss_scale
        self.up_scale_window_ = up_scale_window
        self.min_loss_scale_ = min_loss_scale
        self.max_loss_scale_ = max_loss_scale
        self.loss_scale_ = loss_scale
        self.stable_steps_ = 0

    def update_loss_scale(self, is_all_finite):
        if not self.is_dynamic_scale_:
            return

        if is_all_finite:
            self.stable_steps_ += 1

            if self.stable_steps_ >= self.up_scale_window_:
                self.loss_scale_ = min(self.max_loss_scale_, self.loss_scale_ * 2)
                self.stable_steps_ = 0
        else:
            self.loss_scale_ = max(self.min_loss_scale_, self.loss_scale_ / 2)
            self.stable_steps_ = 0

    def reset(self):
        self.loss_scale_ = self.initial_loss_scale_
        self.stable_steps_ = 0

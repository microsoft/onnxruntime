import copy
import io
import logging
import onnx
import onnxruntime
import os
import torch
import warnings
from onnxruntime.capi import _pybind_state as C

from . import _utils


ONNX_OPSET_VERSION = 12


class ORTModule(torch.nn.Module):

    def __init__(self, module):
        assert isinstance(module, torch.nn.Module), "'module' mst be a torch.nn.Module"
        super(ORTModule, self).__init__()

        # User module is wrapped to use its initializers and save computed gradients
        self._original_module = module
        self._original_module_grad_output_len = -1
        self._original_module_forward_input_grads = []
        self._onnx_training = None
        self._onnx_training_inputs_desc = []
        self._onnx_training_outputs_desc = []
        self._onnx_gradient = None
        self._grad_builder_config = C.ModuleGradientGraphBuilderConfiguration()

        # Forward pass
        self._onnx_forward = None
        self._forward_session = None
        self._onnx_forward_initializers_desc = []
        self._onnx_forward_inputs_desc = []
        self._onnx_forward_outputs_desc = []
        self._onnx_forward_intermediate_outputs_desc = []

        # Backward pass
        self._onnx_backward = None
        self._backward_session = None
        self._onnx_backward_initializers_desc = []
        self._onnx_backward_inputs_desc = []
        self._onnx_backward_gradient_inputs_desc = []
        self._onnx_backward_outputs_desc = []

        # Log level
        self._loglevel = getattr(logging, 'WARNING')

        # TODO: debug flags
        self._save_onnx = False
        self._save_onnx_prefix = ''


    def forward(self, *inputs, **kwargs):
        '''Forward pass starts here and continues at `_ORTModuleFunction.forward`

        ONNX model is exported the first time this method is executed.
        Next, a full training graph is splitted in forward and backward graph which are used
        to instantiate ONNX Runtime InferenceSession`s

        TODO: #ImproveGraphSplitting
        Additionally to that,  several descriptor lists are generated to help identify
        model input, output, initializer, intermediate and gradient tensors.
        '''
        if not self._onnx_forward:
            self._onnx_training = ORTModule._get_forward_graph(self._original_module, *inputs, **kwargs)
            self._onnx_gradient, self._onnx_forward, self._onnx_backward = ORTModule._build_gradient_graph(self._onnx_training, self._grad_builder_config)

            if self._save_onnx:
                onnx.save(self._onnx_training, self._save_onnx_prefix + '_full_training.onnx')
                onnx.save(self._onnx_gradient, self._save_onnx_prefix + '_with_grad.onnx')
                onnx.save(self._onnx_forward, self._save_onnx_prefix + '_forward.onnx')
                onnx.save(self._onnx_backward, self._save_onnx_prefix + '_backward.onnx')

            # TODO: hard-coding to CPU only
            self._forward_session = onnxruntime.InferenceSession(self._onnx_forward.SerializeToString(), providers=['CPUExecutionProvider'])
            self._backward_session = onnxruntime.InferenceSession(self._onnx_backward.SerializeToString(), providers=['CPUExecutionProvider'])

        # Forward I/O description
        if not self._onnx_training_inputs_desc:
            self._onnx_training_inputs_desc = self._get_input_from_graph(self._onnx_training)
            logging.debug(f'Training inputs:\n\t {self._onnx_training_inputs_desc}')
        if not self._onnx_training_outputs_desc:
            self._onnx_training_outputs_desc = self._get_output_from_graph(self._onnx_training)
            logging.debug(f'Training outputs:\n\t {self._onnx_training_outputs_desc}')
        if not self._onnx_forward_initializers_desc:
            self._onnx_forward_initializers_desc = self._get_initializer_from_graph(self._onnx_forward)
            logging.debug(f'Forward initializers:\n\t {self._onnx_forward_initializers_desc}')
        if not self._onnx_forward_inputs_desc:
            self._onnx_forward_inputs_desc = self._get_input_from_graph(self._onnx_forward)
            logging.debug(f'Forward inputs:\n\t {self._onnx_forward_inputs_desc}')
        if not self._onnx_forward_outputs_desc:
            self._onnx_forward_outputs_desc = self._get_output_from_graph(self._onnx_forward)
            logging.debug(f'Forward outputs:\n\t {self._onnx_forward_outputs_desc}')
        if not self._onnx_forward_intermediate_outputs_desc:
            self._onnx_forward_intermediate_outputs_desc = self._get_intermediate_from_forward_graph(self._onnx_forward)
            logging.debug(f'Forward intermediate outputs:\n\t {self._onnx_forward_intermediate_outputs_desc}')

        # Backward I/O description
        if not self._onnx_backward_initializers_desc:
            self._onnx_backward_initializers_desc = self._get_input_from_graph(self._onnx_backward, True)
            logging.debug(f'Backward initializers: {self._onnx_backward_initializers_desc}')
        if not self._onnx_backward_inputs_desc:
            self._onnx_backward_inputs_desc = self._get_input_from_graph(self._onnx_backward, False, self._onnx_backward_initializers_desc)
            logging.debug(f'Backward inputs: {self._onnx_backward_inputs_desc}')
        if not self._onnx_backward_gradient_inputs_desc:
            self._onnx_backward_gradient_inputs_desc = self._get_gradient_input_from_graph(self._onnx_backward, self._onnx_forward_inputs_desc, self._onnx_forward_initializers_desc, self._onnx_forward_intermediate_outputs_desc)
            logging.debug(f'Backward gradient inputs: {self._onnx_backward_gradient_inputs_desc}')
        if not self._onnx_backward_outputs_desc:
            self._onnx_backward_outputs_desc = self._get_output_from_graph(self._onnx_backward)
            logging.debug(f'Backward outputs: {self._onnx_backward_outputs_desc}')

        # Use a custom torch.autograd.Function to associate self.backward_graph as the
        # gradient implementation for self.forward_graph.
        class _ORTModuleFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *inputs, **kwargs):
                '''Performs forward pass based on user input and PyTorch initializer

                TODO: **kwargs are not supported

                Model outputs are returned to the user
                The following tensors are stashed (in order) for backward pass
                    * (Partial) user input
                    * (Partial) Initializers
                    * Intermediate tensors

                TODO: #ImproveGraphSplitting
                String matching to separate user input from initializer
                '''

                # Convert input to dict of torch tensors
                data_dict = self._convert_forward_input_list_to_dict(*inputs)

                # Convert dict of torch tensors to dict of numpy arrays (ORT BE requirement)
                data_dict_numpy = self._convert_dict_torch_to_numpy(data_dict)

                # Feed forward
                outputs, intermediate = self._run_forward_graph(data_dict_numpy)
                outputs = tuple(torch.from_numpy(item) for item in outputs)

                # Save input, initializers and intermediate tensors to be used during backward
                initializer_names = [item['name'] for item in self._onnx_backward_initializers_desc]
                input_names = [item['name'] for item in self._onnx_backward_inputs_desc if item['name'] not in initializer_names]
                ctx_input = tuple(v for k,v in data_dict.items() if k in input_names)
                ctx_initializer = tuple(v for k,v in data_dict.items() if k in initializer_names)
                intermediate = tuple(torch.from_numpy(item) for item in intermediate)
                ctx.save_for_backward(*[*ctx_input, *ctx_initializer, *intermediate])

                # TODO: Support original module output (currently dict is not supported)
                if len(outputs) == 1:
                    return outputs[0]
                return outputs

            @staticmethod
            def backward(ctx, *grad_output):
                '''Performs backward pass based on grad wrt output and internal state

                Internal state is composed of:
                    * Tensor stashed (in a particular order) during forward:
                        * (partial) user input, (partial) initializers and intermediate tensors

                TODO: #ImproveGraphSplitting
                Length of `*grad_output` is needed to detect intermediate tensors during backward pass

                TODO: Input gradient is hard-coded to torch.tensor([1.])
                '''
                saved_tensors = ctx.saved_tensors
                # Used to create backward input
                if self._original_module_grad_output_len == -1:
                    self._original_module_grad_output_len = len(grad_output)

                grad_weights = self._run_backward_graph(*[*saved_tensors, *grad_output])

                result = [torch.tensor([1])]* len(self._onnx_training_inputs_desc)
                result += [torch.from_numpy(grad) for grad in grad_weights]
                return tuple(result)

        return _ORTModuleFunction.apply(*self._convert_forward_input_to_list(*inputs, **kwargs))

    def _convert_forward_input_to_list(self, *inputs, **kwargs):
        '''Creates forward `*inputs` list from user input and PyTorch initializers

        TODO: **kwargs is not supported

        ONNX Runtime forward requires an order list of:
            * User input: computed from forward InferenceSession
            * Initializers: computed from original PyTorch model parameters
        '''

        # List containing both user inputs and initializers, in this order
        result = []

        # Inputs
        for idx, input_data in enumerate(self._forward_session.get_inputs()):
            result.append(inputs[idx])

        # Initializers
        for idx, param in enumerate(self._original_module.named_parameters()):
            result.append(param[1])

        return result

    def _convert_dict_torch_to_numpy(self, tensor_dict):
        '''Convert `tensor_dict` PyTorch tensors to numpy tensors

        This is a ONNX Runtime requirement

        TODO: #UseIOBinding
        '''
        result = {}
        for k,v in tensor_dict.items():
            result.update({k : v.detach().cpu().numpy()})
        return result

    def _convert_forward_input_list_to_dict(self, *inputs):
        '''Convert forward `*inputs` list to dict

        TODO: #ImproveGraphSplitting
        Additionally, a list of gradient names of initializers are created to be used by backprop

        TODO: Input gradient is being ignored for MVP
        '''

        # Dictionary containing both inputs and initializers
        result = {}

        # Inputs
        result_len = 0
        for idx, input_data in enumerate(self._forward_session.get_inputs()):
            result_len += 1
            result.update({input_data.name: inputs[idx]})

        # Initializers
        for param in self._original_module.named_parameters():
            result.update({param[0]: inputs[result_len]})
            # TODO: Create order list of input grads to use during backward.
            #       (for scenarios where gradients of input is required - not covered on MVP)
            # if len(self._original_module_forward_input_grads) < len(self._onnx_training_inputs_desc):
            #     self._original_module_forward_input_grads.append(param[0]+'_grad')

            # TODO: Create order list of initializer grads to use during backward.
            # if len(self._original_module_forward_input_grads) < len(self._onnx_backward_outputs_desc) + len(self._onnx_training_inputs_desc):
            if len(self._original_module_forward_input_grads) < len(self._onnx_backward_outputs_desc):
                self._original_module_forward_input_grads.append(param[0]+'_grad')
            result_len += 1

        return result

    def _convert_backward_input_list_to_dict(self, *inputs):
        '''Convert backward `*inputs` list to dict

        ONNX Runtime backend requires dict as input, which is composed of:
            * User input
                Although not necessary, all user inputs are used for simplicity
            * (Partial) Initializers
                    init_begin = len(user_input)
                    init_count = len(Pre-computed list of initializer)
            * Intermediate tensors TODO: #ImproveGraphSplitting
                Intermediate tensors are inferred from input position:
                    interm_begin = len(user_input) + len(initializer)
                    interm_count = len(all_inputs) - len(user_input) - len(initializer) - len(grad_output)
            * Gradient wrt outputs TODO: #ImproveGraphSplitting
                Gradient tensors are inferred from input position:
                    grads_begin = len(user_input) + len(initializer) + len(intermediate)
                    grads_count = len(all_inputs) - len(user_input) - len(initializer) - len(intermediate)
        '''

        # Dictionary containing both inputs and initializers
        result = {}

        # Inputs
        result_len = 0
        for idx, input_data in enumerate(self._forward_session.get_inputs()):
            result.update({ input_data.name : inputs[idx]})
            result_len += 1

        # Initializers
        for initializer in self._onnx_backward_initializers_desc:
            result.update({initializer['name']: inputs[result_len]})
            result_len += 1

        # Intermediate
        intermediate_len = len(inputs) - result_len - self._original_module_grad_output_len
        for idx in range(intermediate_len):
            result.update({self._onnx_forward_intermediate_outputs_desc[idx]['name']: inputs[result_len]})
            result_len += 1

        # Grad outputs
        for idx in range(len(inputs)-result_len):
            result.update({self._onnx_backward_gradient_inputs_desc[idx]['name']: inputs[result_len]})
            result_len += 1

        return result

    def _run_forward_graph(self, inputs):
        '''Execute forward pass on ONNX Runtime

        Output order has to be specified to ONNX Runtime backend
            to distinguish intermediate from output tensors
        '''

        output_names = [out['name'] for out in self._onnx_forward_outputs_desc]
        forward_output = self._forward_session.run(output_names, inputs)
        output = forward_output[:len(self._onnx_training_outputs_desc)]
        intermediates = forward_output[len(self._onnx_training_outputs_desc):]
        return output, intermediates

    def _run_backward_graph(self, *inputs, **kwargs):
        '''Execute backward pass on ONNX Runtime

        `*inputs` is converted from list to a list of detached numpy tensors before
        being fed to an ONNX Runtime InferenceSession

        TODO: **kwargs are not supported
        '''

        # Convert input to dict of torch tensors
        data = self._convert_backward_input_list_to_dict(*inputs)

        # Convert dict of torch tensors to dict of numpy arrays (ORT BE requirement)
        data = self._convert_dict_torch_to_numpy(data)
        return self._backward_session.run(self._original_module_forward_input_grads, data)

    @staticmethod
    def _get_forward_graph(module, *inputs, **kwargs):
        '''Exports PyTorch `module` to ONNX with training flag, using `*inputs` as input

        TODO: How to support dynamic axes? Dimensions are determined by samples
        TODO: How to ingest **kwargs in proper order during export?
        '''
        # Export the model to memory
        f = io.BytesIO()

        # Deepcopy inputs, since input values may change after model run.
        sample_inputs_copy = copy.deepcopy(inputs)

        # TODO: Support contrib OPs support? user model has no hint
        # from onnxruntime.training import register_custom_ops_pytorch_exporter
        # register_custom_ops_pytorch_exporter.register_custom_op()

        # Export torch.nn.Module to ONNX
        torch.onnx.export(module,
                          tuple(sample_inputs_copy),
                          f,
                          opset_version=ONNX_OPSET_VERSION,
                          do_constant_folding=False,
                          training=torch.onnx.TrainingMode.TRAINING)
        return onnx.load_model_from_string(f.getvalue())

    def _get_initializer_from_graph(self, graph):
        '''Returns a descriptor list of initializers for `graph`

        The list descriptor has the following format:
            [{ 'name': name, 'shape':[int1,...,intN], 'dtype': <onnx.dtype> ]}]

        For ONNX types, refer to https://github.com/onnx/onnx/blob/master/onnx/onnx.in.proto#L461
        '''

        # TODO: There is a tradeoff between memory footprint and total model export time
        #   Ideally we want to export the model using torch.onnx.export(.., export_params=False, keep_initializers_as_inputs=True)
        #   to obtain an ONNX model with minimal size and initializers as input.
        #   However, this results in (guessing) assuming only initializer's name end with '.weight' and '.bias'.
        #   Otherwise, it is not possible to separate input from initializer after the model is exported
        #   Options are:
        #   1) If memory footprint is more important, we can export ONNX twice, varying keep_initializers_as_inputs flag
        #       ONNX model is small (400 bytes vs 1.6MB for MNIST), but export takes twice the time
        #   2) If total export time is more important, we can export ONNX once, using export_params=True
        #       ONNX model is bigger, but export takes half the time

        # As performance is not the main goal in this first deliverable, using approach 2) for simplicity
        initializers = []
        for initializer in graph.graph.initializer:
            name = initializer.name
            shape = initializer.dims
            dtype = _utils.dtype_onnx_to_torch(initializer.data_type)
            initializers.append({'name': name, 'shape': shape, 'dtype': dtype})
        return initializers

    def _get_input_from_graph(self, graph, initializers_only=False, append_initializers=[]):
        '''Returns a descriptor list of input tensors for an ONNX `graph`

        When `initializers_only=True`, only input initializers are returned. Otherwise, both
        user input and initializers are considered.
        This is being used to get backward initializer list TODO: #ImproveGraphSplitting

        When `append_initializers` is not empty, this list is appended to the end of the result list
        This is being used to get backward input list TODO: #ImproveGraphSplitting

        The list descriptor has the following format:
            [{ 'name': name, 'shape':[int1,...,intN], 'dtype': <onnx.dtype> ]}]

        For ONNX types, refer to https://github.com/onnx/onnx/blob/master/onnx/onnx.in.proto#L461
        '''

        inputs = []
        for elem in graph.graph.input:
            for initializer in self._onnx_forward_initializers_desc:
                if elem.name == initializer['name']:
                    if initializers_only:
                        name = elem.name
                        shape = [dim.dim_value for dim in elem.type.tensor_type.shape.dim]
                        dtype = _utils.dtype_onnx_to_torch(elem.type.tensor_type.elem_type)
                        inputs.append({'name': name, 'shape': shape, 'dtype': dtype})
                    break
            else:
                if not initializers_only:
                    name = elem.name
                    shape = [dim.dim_value for dim in elem.type.tensor_type.shape.dim]
                    dtype = _utils.dtype_onnx_to_torch(elem.type.tensor_type.elem_type)
                    inputs.append({'name': name, 'shape': shape, 'dtype': dtype})
        if append_initializers:
            inputs.extend(append_initializers)
        return inputs

    def _get_gradient_input_from_graph(self, backward_graph, forward_input, forward_initializer, forward_intermediate):
        '''Returns a descriptor list of gradient output for `backward_graph`

        Gradient output tensors are found through an elimination process, that cross reference
        inputs from the backward graph to the forward input, initializer and intermediate tensors.

        The list descriptor has the following format:
            [{ 'name': name, 'shape':[int1,...,intN], 'dtype': <onnx.dtype> ]}]

        For ONNX types, refer to https://github.com/onnx/onnx/blob/master/onnx/onnx.in.proto#L461

        TODO: #ImproveGraphSplitting
        '''
        grads = []
        found = False
        for elem in backward_graph.graph.input:
            for item in forward_input:
                if elem.name == item['name']:
                    # skip output
                    break
            else:
                for item in forward_initializer:
                    if elem.name == item['name']:
                        # skip output
                        break
                else:
                    for item in forward_intermediate:
                        if elem.name == item['name']:
                            # skip output
                            break
                    else:
                        name = elem.name
                        shape = [dim.dim_value for dim in elem.type.tensor_type.shape.dim]
                        dtype = _utils.dtype_onnx_to_torch(elem.type.tensor_type.elem_type)
                        grads.append({'name': name, 'shape': shape, 'dtype': dtype})
        return grads

    def _get_output_from_graph(self, graph):
        '''Returns a descriptor list of output tensors for an ONNX `graph`

        The list descriptor has the following format:
            [{ 'name': name, 'shape':[int1,...,intN], 'dtype': <onnx.dtype> ]}]

        For ONNX types, refer to https://github.com/onnx/onnx/blob/master/onnx/onnx.in.proto#L461
        '''
        outputs = []
        for elem in graph.graph.output:
            for initializer in self._onnx_forward_initializers_desc:
                if elem.name == initializer['name']:
                    # skip initializers
                    break
            else:
                name = elem.name
                shape = [dim.dim_value for dim in elem.type.tensor_type.shape.dim]
                dtype = _utils.dtype_onnx_to_torch(elem.type.tensor_type.elem_type)
                outputs.append({'name': name, 'shape': shape, 'dtype': dtype})
        return outputs

    def _get_intermediate_from_forward_graph(self, forward_graph):
        '''Returns a descriptor list with all intermediate tensors for `forward_graph`

        Intermediate tensors are found through an elimination process, that cross reference
        outputs from the forward graph to the original model (exported to ONNX)

        The list descriptor has the following format:
            [{ 'name': name, 'shape':[int1,...,intN], 'dtype': <onnx.dtype> ]}]

        TODO: #ImproveGraphSplitting
        '''
        intermediates = []
        for elem in forward_graph.graph.output:
            for output in self._onnx_training_outputs_desc:
                if elem.name == output['name']:
                    # skip output
                    break
            else:
                name = elem.name
                shape = [dim.dim_value for dim in elem.type.tensor_type.shape.dim]
                dtype = _utils.dtype_onnx_to_torch(elem.type.tensor_type.elem_type)
                intermediates.append({'name': name, 'shape': shape, 'dtype': dtype})
        return intermediates

    @staticmethod
    def _build_gradient_graph(forward_graph, config):
        '''Adds gradient nodes on top of an existing ONNX graph (with training flag)

        TODO: #SplittingGraphAtFrontend
        '''
        if not config.weight_names_to_train:
            weight_names_to_train = set()
            for initializer in forward_graph.graph.initializer:
                weight_names_to_train.add(initializer.name)
            config.weight_names_to_train = weight_names_to_train
            output_names = set()
            for output in forward_graph.graph.output:
                output_names.add(output.name)
            config.output_names = output_names
        models = [onnx.load_model_from_string(model_as_string)
                  for model_as_string in C.ModuleGradientGraphBuilder().build_and_split(forward_graph.SerializeToString(), config)]
        return models[0], models[1], models[2]

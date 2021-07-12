from typing import Sequence
import torch

from ._execution_agent import TrainingAgent

def ort_module_function_factory(execution_agent: TrainingAgent, user_input_names: Sequence[str], require_grad_names: Sequence[str],
                                initializer_names: Sequence[str], initializer_names_to_train: Sequence[str],
                                module_output_indices_requires_save_for_backward: Sequence[int] = ()) -> torch.autograd.Function:
    """ Creates a torch.autograd.Function for ORTModule.

    Args:
        execution_agent: TrainingAgent responsible for performing the forward / backward computations
        user_input_names: Sequence of tensor names corresponding to forward method arguments
        require_grad_names: Subset of user_input_names that require gradients
        initializer_names: Sequence of tensor names corresponding to torch parameters
        initializer_names_to_train: Subset of initializer_names that require gradients
        module_output_indices_requires_save_for_backward: Sequence of indices corresponding to output tensors required in backward computation

    Returns:
        a torch.autograd.Function class that implements forward / backward computations using onnxruntime.
    """

    class _ORTModuleFunction(torch.autograd.Function):
        '''Use a custom torch.autograd.Function to associate self.backward_graph as the
        gradient implementation for self.forward_graph.'''

        @staticmethod
        def forward(ctx, *inputs):
            '''Performs forward pass based on user input and PyTorch initializer

            Autograd Function's apply() doesn't support keyword arguments,
            so `*inputs` has all the arguments - keyword arguments converted
            to positional/keywords during `TrainingManager.forward`.

            Module outputs are returned to the user
            '''

            user_outputs, ctx.run_info = execution_agent.forward(*inputs)

            # Disable materializing grads then None object will not be
            # converted to a tensor filled with zeros prior to calling backward.
            # Save shape, device and type info to ctx for materializing tensor in backward if output grad is None.
            ctx.set_materialize_grads(False)

            # Mark the outputs tensors needed in backward computation
            # ORT is NOT relying on save_for_backward() to actually save the tensor, 
            # as this tensor is also kept in ORT's PartialGraphState
            # This call is to invoke pytorch's version check to detect the potential inplace corruption
            for idx in module_output_indices_requires_save_for_backward:
                ctx.save_for_backward(user_outputs[idx])

            return user_outputs

        @staticmethod
        def backward(ctx, *grad_outputs):
            '''Performs backward pass based on grad wrt module output'''

            assert ctx.run_info is not None, 'forward() or __call__() methods must be called before backward()'

            # Unpack saved_tensor to trigger version detection that catches inplace corruption
            _ = ctx.saved_tensors

            backward_outputs = execution_agent.backward(ctx.run_info, *grad_outputs)

            # Destroy the state immediately (as opposed to be at the mercy of garbage collector) so it does not
            # affect peak memory usage in a subsequent graph run.
            del ctx.run_info.state
            # Return input and initializer gradients
            num_user_input_grads = len(require_grad_names)
            results = []
            require_grad_names_set = set(require_grad_names)
            require_grad_names_index = 0
            for input_name in user_input_names:
                # Append to the results the backward output for each input that required grad
                if input_name in require_grad_names_set:
                    results.append(backward_outputs[require_grad_names_index])
                    require_grad_names_index += 1
                else:
                    # input_name is not found in the self._input_info.require_grad_names list
                    # Append None to results for each input that did not require grad
                    results.append(None)
            assert require_grad_names_index == num_user_input_grads
            # Append gradients of initializer to results
            # Go over each initializer, check if it required grad and append to results accordingly
            initializer_index = num_user_input_grads
            for initializer_name in initializer_names:
                if initializer_name in initializer_names_to_train:
                    results.append(backward_outputs[initializer_index])
                    initializer_index += 1
                else:
                    results.append(None)
            assert initializer_index == len(backward_outputs)
            return tuple(results)

    return _ORTModuleFunction

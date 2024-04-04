# ORTModule Custom Autograd Function Support

## What is autograd Functions?

`PyTorch` allows users to define customized operators (for its forward and backward implementations) [PyTorch: Defining New autograd Functions](https://github.com/pytorch/tutorials/blob/d98606855d3c8c5bd78d55b95717be5a02960363/beginner_source/examples_autograd/polynomial_custom_function.py#L25).

There are many such use cases as more optimized deep learning projects keep growing, here we just name a few:
- [NVIDIA/apex](https://github.com/NVIDIA/apex/blob/58acf96915eecd7e13adff61d2c389fba3efede2/apex/transformer/functional/fused_softmax.py#L21)
- [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM/blob/f7727433293427bef04858f67b2889fe9b177d88/megatron/core/tensor_parallel/mappings.py#L220C31-L220C31)
- [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention/blob/3a9fe7b0faaa9d648394026c9c20231c07bf999d/flash_attn/flash_attn_interface.py#L429),
- [openai/triton](https://github.com/openai/triton/blob/424e67e7275f0cb2cd231e7a4d17ff8570530b77/python/tutorials/06-fused-attention.py#L457)
- ...

Those operators are used in training/evaluation scenarios a lot, where is ORTModule capability overlaps.
To best release ORTModule's acceleration power, we need tolerant and handle those customized operators
from the to-onnx conversion, to backward graph building, and also its execution in runtime as a full lifecycle.

## How ORTModule support autograd.Function?

The way we have here is through introduced `PythonOp`/`PythonOpGrad` MS domain operators in `ONNX Runtime`,
- Map autograd Function (`prim::PythonOp` in `PyTorch`) to `PythonOp` in `ONNX Runtime` during model export by [registering customized export function](https://github.com/microsoft/onnxruntime/blob/c2bd5b70b29eb3c687c5497696e7b0a1930604d3/orttraining/orttraining/python/training/ortmodule/_custom_autograd_function.py#L69C16-L69C16)
  ```
    class ScalarAndTupleFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha, beta, gamma):
            ctx.save_for_backward(input)
            ctx.alpha = alpha
            ctx.beta = beta
            ctx.gamma = gamma
            return alpha * beta[0] * beta[1] * gamma * input.clamp(min=0)

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            alpha = ctx.alpha
            beta = ctx.beta
            gamma = ctx.gamma
            grad_input = grad_output.clone()
            grad_input[input < 0] = 0
            return alpha * beta[0] * beta[1] * gamma * grad_input, None, None, None
  ```
  The example above shows a customized function taking 4 inputs (despite of ctx), the first input is a tensor [exporter treats it as input for `PythonOp`](https://github.com/microsoft/onnxruntime/blob/c2bd5b70b29eb3c687c5497696e7b0a1930604d3/orttraining/orttraining/python/training/ortmodule/_custom_autograd_function_exporter.py#L174),
  the others are scalars, export function will convert all such non-tensor inputs to constant and [stores
  in `PythonOp`'s attributes](https://github.com/microsoft/onnxruntime/blob/c2bd5b70b29eb3c687c5497696e7b0a1930604d3/orttraining/orttraining/python/training/ortmodule/_custom_autograd_function_exporter.py#L272). Things to be noted here: if the non-tensor
  input is one of those types "bool scalar, int scalar, float scalar, bool tuple, int tuple, float tuple", they will be
  stored in corresponding attributes; otherwise, they will be treated a `object` and the object address stored in `input_pointer_scalars` ([reference count will be increased](https://github.com/microsoft/onnxruntime/blob/c2bd5b70b29eb3c687c5497696e7b0a1930604d3/orttraining/orttraining/python/training/ortmodule/_custom_autograd_function_exporter.py#L250C27-L250C27) also to make sure it exists during model run).
- [PythonOp kernel](https://github.com/microsoft/onnxruntime/blob/c2bd5b70b29eb3c687c5497696e7b0a1930604d3/orttraining/orttraining/training_ops/cuda/torch/torch_custom_function_kernel.cc#L38) is responsible to run the `forward` interface user defined through [forward runner](https://github.com/microsoft/onnxruntime/blob/c2bd5b70b29eb3c687c5497696e7b0a1930604d3/orttraining/orttraining/python/training/ortmodule/_custom_autograd_function_runner.py#L409).
Similarly, [PythonOpGrad kernel](https://github.com/microsoft/onnxruntime/blob/c2bd5b70b29eb3c687c5497696e7b0a1930604d3/orttraining/orttraining/training_ops/cuda/torch/torch_custom_function_kernel.cc#L49) is responsible to run the `backward` interface user defined through [backward runner](https://github.com/microsoft/onnxruntime/blob/c2bd5b70b29eb3c687c5497696e7b0a1930604d3/orttraining/orttraining/python/training/ortmodule/_custom_autograd_function_runner.py#L554).

Currently, for training python wheel, `PythonOp` support is by default enabled, users don't need to be aware of it. As long as the
defined torch.autograd.Function is working in `PyTorch` run, it should be runnable with `ORTModule`. If you need to enable it or
disable it explicitly, refer to the [wiki](https://github.com/microsoft/onnxruntime/blob/main/docs/ORTModule_Training_Guidelines.md#ortmodule_enable_custom_autograd).



## Known Issues and Workaround

PyTorch Versions
- Minimum version 1.9 (introduced "Support registering custom export for `prim::PythonOp`` from torch.autograd.Function ([#55630](https://github.com/pytorch/pytorch/pull/55630)) ([#57600](https://github.com/pytorch/pytorch/pull/57600))")
- If the static forward function has only one output, any version of Pytorch 1.9 is fine. Otherwise, a PyTorch version containing [this commit](https://github.com/pytorch/pytorch/commit/a55cae3d37e0f7852e391886c3904307caa4d06d) is required.
- [Throw _Map_base::at Exception](https://github.com/pytorch/pytorch/issues/88286), export errors like this:
  ```
	RuntimeError: There was an error while exporting the PyTorch model to ONNX:

	Traceback (most recent call last):
	File "/opt/conda/envs/ptca/lib/python3.8/site-packages/onnxruntime/training/ortmodule/_utils.py", line 316, in get_exception_as_string
		raise exception
	File "/opt/conda/envs/ptca/lib/python3.8/site-packages/onnxruntime/training/ortmodule/_graph_execution_manager.py", line 425, in _get_exported_model
		torch.onnx.export(
	File "/opt/conda/envs/ptca/lib/python3.8/site-packages/torch/onnx/utils.py", line 506, in export
		_export(
	File "/opt/conda/envs/ptca/lib/python3.8/site-packages/torch/onnx/utils.py", line 1548, in _export
		graph, params_dict, torch_out = _model_to_graph(
	File "/opt/conda/envs/ptca/lib/python3.8/site-packages/torch/onnx/utils.py", line 1113, in _model_to_graph
		graph, params, torch_out, module = _create_jit_graph(model, args)
	File "/opt/conda/envs/ptca/lib/python3.8/site-packages/torch/onnx/utils.py", line 989, in _create_jit_graph
		graph, torch_out = _trace_and_get_graph_from_model(model, args)
	File "/opt/conda/envs/ptca/lib/python3.8/site-packages/torch/onnx/utils.py", line 893, in _trace_and_get_graph_from_model
		trace_graph, torch_out, inputs_states = torch.jit._get_trace_graph(
	File "/opt/conda/envs/ptca/lib/python3.8/site-packages/torch/jit/_trace.py", line 1268, in _get_trace_graph
		outs = ONNXTracedModule(f, strict, _force_outplace, return_inputs, _return_inputs_states)(*args, **kwargs)
	...
	File "/opt/conda/envs/ptca/lib/python3.8/site-packages/deepspeed-0.9.5+95680ca-py3.8.egg/deepspeed/runtime/zero/parameter_offload.py", line 632, in _ort_post_forward_module_hook
		a = ORTPostForwardwardFunction.apply(module, _post_forward_module_hook, _ort_run_before_backward_function, len(input), len(output), *input_and_output)
	File "/opt/conda/envs/ptca/lib/python3.8/site-packages/torch/autograd/function.py", line 506, in apply
		return super().apply(*args, **kwargs)  # type: ignore[misc]
	RuntimeError: _Map_base::at
  ```
  Resolution: upgrade `PyTorch` to new versions containing [this commit](https://github.com/thiagocrepaldi/pytorch/commit/3d3da109e3afa617c513e78aa999f5a1f44ffbce), when export param `autograd_inlining` is [set to false](https://github.com/microsoft/onnxruntime/blob/0e2782438a65b97919f15af14d2a4ada361157b6/orttraining/orttraining/python/training/ortmodule/_graph_execution_manager.py#L387C26-L387C26) to skip this error.
- "Tried to trace <__torch__.torch.classes.c10d.ProcessGroup object at 0x2969c520> but it is not part of the active trace"
   This usually happens when torch.autograd.Function's forward function used `PyTorch` collective calls and pass the group explicitly.
  ```
	RuntimeError: There was an error while exporting the PyTorch model to ONNX:

	Traceback (most recent call last):
	File "/bert_ort/pengwa/py3.8/lib/python3.8/site-packages/onnxruntime/training/ortmodule/_utils.py", line 324, in get_exception_as_string
		raise exception
	File "/bert_ort/pengwa/py3.8/lib/python3.8/site-packages/onnxruntime/training/ortmodule/_graph_execution_manager.py", line 342, in _get_exported_model
		torch.onnx.export(
	File "/bert_ort/pengwa/py3.8/lib/python3.8/site-packages/torch/onnx/utils.py", line 507, in export
		_export(
	File "/bert_ort/pengwa/py3.8/lib/python3.8/site-packages/torch/onnx/utils.py", line 1567, in _export
		graph, params_dict, torch_out = _model_to_graph(
	File "/bert_ort/pengwa/py3.8/lib/python3.8/site-packages/torch/onnx/utils.py", line 1124, in _model_to_graph
		graph, params, torch_out, module = _create_jit_graph(model, args)
	File "/bert_ort/pengwa/py3.8/lib/python3.8/site-packages/torch/onnx/utils.py", line 1000, in _create_jit_graph
		graph, torch_out = _trace_and_get_graph_from_model(model, args)
	File "/bert_ort/pengwa/py3.8/lib/python3.8/site-packages/torch/onnx/utils.py", line 904, in _trace_and_get_graph_from_model
		trace_graph, torch_out, inputs_states = torch.jit._get_trace_graph(
	File "/bert_ort/pengwa/py3.8/lib/python3.8/site-packages/torch/jit/_trace.py", line 1269, in _get_trace_graph
		outs = ONNXTracedModule(f, strict, _force_outplace, return_inputs, _return_inputs_states)(*args, **kwargs)
	File "/bert_ort/pengwa/py3.8/lib/python3.8/site-packages/torch/jit/_trace.py", line 128, in forward
		graph, out = torch._C._create_graph_by_tracing(
    ...
	File "/bert_ort/pengwa/deepspeed/deepspeed/runtime/zero/parameter_offload.py", line 640, in _ort_pre_forward_module_hook
		rets = ORTPreForwardwardFunction.apply(self, module, _ort_run_after_backward_function, *inputs)
	...
	File "/bert_ort/pengwa/deepspeed/deepspeed/runtime/zero/parameter_offload.py", line 823, in pre_sub_module_forward_function
		param_coordinator.fetch_sub_module(sub_module, forward=True)
	...
	File "/bert_ort/pengwa/py3.8/lib/python3.8/site-packages/torch/distributed/distributed_c10d.py", line 2841, in all_gather_into_tensor
		work = group._allgather_base(output_tensor, input_tensor)
	RuntimeError: Tried to trace <__torch__.torch.classes.c10d.ProcessGroup object at 0x56250ad114a0> but it is not part of the active trace. Modules that are called during a trace must be registered as submodules of the thing being traced.
  ```
  Resolution: modify the autograd.Function, to skip the run the collection operator during onnx export, here is an example.
  ```python
    # Pre
	def allgather_fn(output_tensor, input_tensor, group=None, async_op=False, debug=get_caller_func()):
		return torch.distributed.all_gather_into_tensor(output_tensor, input_tensor, group=group, async_op=async_op, debug=debug)

	# Workaround
	from typing import Any, List
	class DummyWork(torch.distributed.distributed_c10d.Work):
		def is_completed(self) -> bool:
			return True
		def is_success(self) -> bool:
			return True
		def exception(self) -> Any:
			return None
		def wait(self, timeout: timedelta = timedelta) -> bool:
			return True
		def source_rank(self) -> int:
			return 0
		def _source_rank(self) -> int:
			return 0
		def result(self) -> List[torch.Tensor]:
			return []
		def synchronize(self):
			pass

	def allgather_fn(output_tensor, input_tensor, group=None, async_op=False, debug=get_caller_func()):
		if torch.onnx.is_in_onnx_export():
			return DummyWork()

		return torch.distributed.all_gather_into_tensor(output_tensor, input_tensor, group=group, async_op=async_op, debug=debug)
  ```

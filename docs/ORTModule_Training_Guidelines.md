# ONNX Runtime Training Guidelines

## Use `ORTModule` to Accelerate Forward/Backward

Plug in your `torch.nn.Module` model with `ORTModule` to leverage ONNX Runtime fast training engine.

### Insallation and Usage

Refer to (https://onnxruntime.ai/)[https://onnxruntime.ai/] to download training wheel.

Sample usage as below:
```diff
	model = build_model()

+	from onnxruntime.training import ORTModule
+	model = ORTModule(model)
```

More options for **developers**.
```diff
	model = build_model()

+	from onnxruntime.training import ORTModule, DebugOptions, LogLevel
+	model = ORTModule(model, DebugOptions(save_onnx=True, log_level=LogLevel.VERBOSE))
```
Check [DebugOptions implementation](../orttraining/orttraining/python/training/ortmodule/debug_options.py) for more details.

### Envrionment Variables

ONNX Runtime Training (ORTModule) provides envrionment variables targeting different use cases.

#### ORTMODULE_ONNX_OPSET_VERSION

- **Feature Area**: *ORTMODULE/ONNXOpset*
- **Description**: By default, as ONNX Runtime released, the ONNX opset version to use will be updated periodically. For some customers, they want to stick to fixed opset where both performance and accuracy are well validated, so this env variable can be used to control that.

	```
	export ORTMODULE_ONNX_OPSET_VERSION=14
	```


#### ORTMODULE_FALLBACK_POLICY

- **Feature Area**: *ORTMODULE/FallbackToPytorch*
- **Description**: By default, if ORTModule failed to run the model using ONNX Runtime backend, it will fallback to use PyTorch to continue the training. At some point developers are optimizing the models and doing benchmarking, we want explicitly let ORT backend to run the model. The way we disable the retry:
	```
	export ORTMODULE_FALLBACK_POLICY="FALLBACK_DISABLE"
	```


#### ORTMODULE_LOG_LEVEL

- **Feature Area**: *ORTMODULE/DebugOptions*
- **Description**: Configure ORTModule log level. Defaults to LogLevel.WARNING.
log_level can also be set by setting the environment variable "ORTMODULE_LOG_LEVEL" to one of "VERBOSE", "INFO", "WARNING", "ERROR", "FATAL". The environment variable takes precedence if DebugOptions also set log_level.

#### ORTMODULE_SAVE_ONNX_PATH

- **Feature Area**: *ORTMODULE/DebugOptions*
- **Description**: Configure ORTModule to save onnx models. Defaults to False.
The output directory of the onnx models by default is set to the current working directory. To change the output directory, the environment variable "ORTMODULE_SAVE_ONNX_PATH" can be set to the destination directory path.


#### ORTMODULE_ALLOW_AUTOGRAD_CHECKPOINT

- **Feature Area**: *ORTMODULE/PythonOp (torch.autograd.Function)*
- **Description**: By default ORTModule will fail with exception when handle PythonOp export for some `'autograd.Function'`s (One example is torch CheckpointFunction). Set
	this env variable to be `1` to explicitly allow it.
	```
	export ORTMODULE_ALLOW_AUTOGRAD_CHECKPOINT=1
	```

	> Take the example of torch.utils.checkpoint.CheckpointFunction, if it is exported as PythonOp, the checkpointed computation may be computed by PyTorch, not ORT. This situation is especially important for big models such as GPT-2 where every few layers are wrapped to do recomputation, large amount of computations are done by PyTorch. Currently a failure is reported to notice users it is possible ORTModule has less opportunaties optimize further.

	> On the other hand, if the wrapped computation graph is small, it is reasonable to allowed it. And users should be aware that ORT performance boost might be trivial.


#### ORTMODULE_DISABLE_CUSTOM_AUTOGRAD_SUPPORT

- **Feature Area**: *ORTMODULE/PythonOp (torch.autograd.Function)*
- **Description**: By default, all torch.autograd.Function classes will be exported to ORT PythonOp. There are some cases you might consider to disable it. For example, if you confirmed those torch.autograd.Function classes defined computations that could be inlined exported by PyTorch, and it is safe to use the inlined exported ONNX graph to train, then you can disable it, as a result, ORT has more opportunaties to optimize more.
	```
	export ORTMODULE_DISABLE_CUSTOM_AUTOGRAD_SUPPORT=1
	```

	An alternative to disable without using envrionment variable:

		```
		from onnxruntime.training.ortmodule._custom_autograd_function import enable_custom_autograd_support
		enable_custom_autograd_support(False)
		```

#### ORTMODULE_SKIPPED_AUTOGRAD_FUNCTIONS

- **Feature Area**: *ORTMODULE/PythonOp (torch.autograd.Function)*
- **Description**: By default, this is empty. When user model's setup depends on libraries who might define multiple torch.autograd.Function classes of same name, though their python import pathes (e.g. 'namespace') are different, while due to limitation of PyTorch exporter (https://github.com/microsoft/onnx-converters-private/issues/115), ORT backend cannot infer which one to call. So an exception will be thrown for this case.

Before full qualified name can be got from exporter, this envrionment variables can be used to specify which torch.autograd.Function classes can be ignored. An example as below, be noted, full qualified name is needed here. If there are multiple classes to be ignored, use comma as the seperator.

	```
	export ORTMODULE_SKIPPED_AUTOGRAD_FUNCTIONS = "orttraining_test_ortmodule_autograd.test_skipped_autograd_function.<locals>.TestSkippedFunction"
	```

### Use `FusedAdam` to Accelerate Parameter Update

Parameter update is done by optimizers (for example AdamW) with many elementwise operations. `FusedAdam` launch the elementwise updates kernels with multi-tensor apply, allowing batches of gradients applied to corresponding models parameters with much less kernel launches.

Here is a sample switch from torch `AdamW` optimizer to `FusedAdam`.

```diff
	model = build_model()

-	adamw_opt = AdamW(model.parameters(), lr=1)

+	from onnxruntime.training.optim import FusedAdam
+	adamw_opt = FusedAdam(model.parameters(), lr=1)

```

Check [FusedAdam implementation](../orttraining/orttraining/python/training/optim/fused_adam.py) for more details.

### Use FP16_Optimizer to Complement DeepSpeed/APEX

If user models utilize DeepSpeed or Apex libraries, ORT's `FP16_Optimizer` can be used to complement some inefficiencies introduced by them.

Use `FP16_Optimizer` with DeepSpeed ZeRO Optimizer:

```diff
	# Could also be ORT's FusedAdam.
	optimizer = AdamW(model.parameters(), lr=1)
	model, optimizer, _, lr_scheduler = deepspeed.initialize(
			model=model,
			optimizer=optimizer,
			args=args,
			lr_scheduler=lr_scheduler,
			mpu=mpu,
			dist_init_required=False)

+	from onnxruntime.training.optim.fp16_optimizer import FP16_Optimizer
+	optimizer = FP16_Optimizer(optimizer)

```

Use `FP16_Optimizer` with Apex Optimizer:
```diff
	optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
	model, optimizer = amp.initialize(model, optimizer, opt_level="O2")

+	from onnxruntime.training.optim.fp16_optimizer import FP16_Optimizer as ORT_FP16_Optimizer
+	optimizer = ORT_FP16_Optimizer(optimizer)

```

Check [FP16_Optimizer implementation](../orttraining/orttraining/python/training/optim/fp16_optimizer.py) for more details.


### Putting All Together `ORTModule` + `FusedAdam` + `FP16_Optimizer`

```diff
	model = build_model()

+	from onnxruntime.training import ORTModule
+	model = ORTModule(model)

-	optimizer = AdamW(model.parameters(), lr=1)
+	from onnxruntime.training.optim import FusedAdam
+	optimizer = FusedAdam(model.parameters(), lr=1)

	model, optimizer, _, lr_scheduler = deepspeed.initialize(
			model=model,
			optimizer=optimizer,
			args=args,
			lr_scheduler=lr_scheduler,
			mpu=mpu,
			dist_init_required=False)

+	from onnxruntime.training.optim.fp16_optimizer import FP16_Optimizer
+	optimizer = FP16_Optimizer(optimizer)

```

### Memory Optimization

Check [Memory Optimizer for ONNX Runtime Training](Memory_Optimizer.md) for more details.

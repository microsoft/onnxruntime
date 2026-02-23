# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import collections
import functools
import inspect
import logging
import multiprocessing
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Union

import onnx
import torch
import transformers
from onnxscript import ir, version_converter
from packaging import version
from transformers.modeling_utils import PreTrainedModel

from olive.common.config_utils import get_the_flattened_and_tree_spec, validate_config
from olive.common.utils import find_submodules, resolve_torch_dtype, tensor_data_to_device, tensor_data_to_dtype
from olive.constants import DiffusersComponent
from olive.hardware import AcceleratorSpec
from olive.model import (
    CompositeModelHandler,
    DiffusersModelHandler,
    DistributedHfModelHandler,
    DistributedOnnxModelHandler,
    HfModelHandler,
    ONNXModelHandler,
    PyTorchModelHandler,
)
from olive.model.config import IoConfig
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, ir_model_to_olive_model
from olive.passes.pass_config import BasePassConfig, PassConfigParam, get_user_script_data_config

# pylint: disable=W0212

logger = logging.getLogger(__name__)


def _torch_is_older_than(version_str: str) -> bool:
    torch_version = version.parse(torch.__version__).release
    return torch_version < version.parse(version_str).release


class TraceModelWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, *input_data, **input_dict):
        if isinstance(self.model(*input_data, **input_dict), dict):
            return list(self.model(*input_data, **input_dict).values())
        return self.model(*input_data, **input_dict)


def _register_dynamic_cache_export_support():
    """Utilities for `DynamicCache` <> torch.export support."""
    from transformers.cache_utils import DynamicCache, DynamicLayer, DynamicSlidingWindowLayer

    def _get_cache_dict(cache: DynamicCache):
        """Convert cache to dictionary format for pytree operations."""
        if any(not isinstance(layer, (DynamicLayer, DynamicSlidingWindowLayer)) for layer in cache.layers):
            raise RuntimeError("This pytree flattening function should only be applied to DynamicCache")

        return {
            "cache": [(layer.keys, layer.values) for layer in cache.layers if layer.keys is not None],
        }

    try:
        torch.utils._pytree.register_pytree_node(
            DynamicCache,
            lambda dynamic_cache: torch.utils._pytree._dict_flatten(_get_cache_dict(dynamic_cache)),
            _unflatten_dynamic_cache,
            serialized_type_name=f"{DynamicCache.__module__}.{DynamicCache.__name__}",
            flatten_with_keys_fn=lambda dynamic_cache: torch.utils._pytree._dict_flatten_with_keys(
                _get_cache_dict(dynamic_cache)
            ),
        )
        # TODO (team): This won't be needed in torch 2.7.
        torch.fx._pytree.register_pytree_flatten_spec(
            DynamicCache,
            lambda cache, spec: torch.fx._pytree._dict_flatten_spec(_get_cache_dict(cache), spec),
        )
    # Catching this in case there are multiple runs for some test runs
    except ValueError as e:
        if "already registered as pytree node" not in str(e):
            raise


def _unflatten_dynamic_cache(values, context: torch.utils._pytree.Context):
    from transformers.cache_utils import DynamicCache

    dictionary = torch.utils._pytree._dict_unflatten(values, context)
    cache = DynamicCache()
    # Reconstruct layers from keys and values lists
    cache_list = dictionary.get("cache", [])
    for i, (key, value) in enumerate(cache_list):
        cache.update(key, value, i)
    return cache


def _patch_dynamic_layer_for_export():
    """Patch DynamicLayer.lazy_initialization for torch.export compatibility (transformers >= 5.0).

    The original uses torch.tensor([]) which creates a 1D empty tensor (shape [0]).
    torch.export needs consistent tensor ranks, so we use torch.narrow + torch.empty_like
    to preserve the full shape (e.g. [batch, heads, 0, head_dim]).
    """
    from transformers.cache_utils import DynamicLayer

    if not hasattr(DynamicLayer, "lazy_initialization"):
        return

    def patched_lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor = None):
        self.dtype, self.device = key_states.dtype, key_states.device
        like = torch.narrow(key_states, dim=-2, start=0, length=0)
        if hasattr(key_states, "fake_mode"):
            with key_states.fake_mode:
                self.keys = torch.empty_like(like, dtype=self.dtype, device=self.device)
                self.values = torch.empty_like(like, dtype=self.dtype, device=self.device)
        else:
            self.keys = torch.empty_like(like, dtype=self.dtype, device=self.device)
            self.values = torch.empty_like(like, dtype=self.dtype, device=self.device)
        self.is_initialized = True

    DynamicLayer.lazy_initialization = patched_lazy_initialization
    logger.debug("Patched DynamicLayer.lazy_initialization for torch.export compatibility.")


def _convert_past_key_values_to_dynamic_cache(dummy_kwargs: dict, config=None) -> dict:
    """Convert legacy list-format past_key_values to DynamicCache (transformers >= 5.0).

    Transformers 5.0 models expect DynamicCache objects, not lists of (key, value) tensors.
    When config is provided, the DynamicCache will create correct layer types (e.g.
    DynamicSlidingWindowLayer for models using sliding window attention).
    """
    pkv = dummy_kwargs.get("past_key_values")
    if pkv is None or not isinstance(pkv, (list, tuple)):
        return dummy_kwargs

    # Check if it's legacy format: list of [key, value] pairs (each with exactly 2 elements)
    if not pkv or not isinstance(pkv[0], (list, tuple)) or len(pkv[0]) != 2:
        return dummy_kwargs

    from transformers.cache_utils import DynamicCache

    dc = DynamicCache(config=config)
    for layer_idx, kv in enumerate(pkv):
        dc.update(kv[0], kv[1], layer_idx=layer_idx)
    dummy_kwargs["past_key_values"] = dc
    logger.debug("Converted past_key_values from legacy list format to DynamicCache.")
    return dummy_kwargs


def _convert_dynamic_shapes_for_dynamic_cache(dynamic_shapes: dict) -> dict:
    """Convert dynamic_shapes for past_key_values from nested list to DynamicCache pytree format.

    The old format is: [[key_shape, val_shape], ...] (one pair per layer)
    The DynamicCache pytree is: {"cache": [(key0, val0), (key1, val1), ...]}
    matching the structure from _register_dynamic_cache_export_support().
    """
    pkv_shapes = dynamic_shapes.get("past_key_values")
    if pkv_shapes is None or not isinstance(pkv_shapes, (list, tuple)):
        return dynamic_shapes

    if not pkv_shapes or not isinstance(pkv_shapes[0], (list, tuple)) or len(pkv_shapes[0]) != 2:
        return dynamic_shapes

    # Convert [[key0, val0], [key1, val1], ...] -> {"cache": [(key0, val0), (key1, val1), ...]}
    # matching DynamicCache pytree: _dict_flatten({"cache": [(keys, values), ...]})
    dynamic_shapes["past_key_values"] = {
        "cache": [tuple(layer) for layer in pkv_shapes],
    }
    logger.debug("Converted dynamic_shapes for past_key_values to DynamicCache pytree format.")
    return dynamic_shapes


def _patch_model_if_necessary(pytorch_model: torch.nn.Module):
    if not isinstance(pytorch_model, PreTrainedModel):
        return

    transformers_version = version.parse(transformers.__version__)
    if transformers_version < version.parse("4.45"):
        return

    orig_forward_name = "forward" if hasattr(pytorch_model, "forward") else "call"
    orig_forward = getattr(pytorch_model, orig_forward_name)
    signature = inspect.signature(orig_forward)

    logits_to_keep_name = "logits_to_keep" if transformers_version >= version.parse("4.49") else "num_logits_to_keep"
    # num_logits_to_keep was added in transformers 4.45 and isn't added as inputs when exporting the model
    logits_to_keep_index = (
        list(signature.parameters.keys()).index(logits_to_keep_name)
        if logits_to_keep_name in signature.parameters
        else None
    )
    pkv_index = (
        list(signature.parameters.keys()).index("past_key_values")
        if "past_key_values" in signature.parameters
        else None
    )

    @functools.wraps(orig_forward)
    def patched_forward(*args, **kwargs):
        from transformers.cache_utils import DynamicCache, EncoderDecoderCache

        args = list(args) if args else []
        kwargs = kwargs or {}

        if logits_to_keep_name in kwargs or (logits_to_keep_index is not None and len(args) <= logits_to_keep_index):
            kwargs[logits_to_keep_name] = 0
        elif logits_to_keep_index is not None:
            args[logits_to_keep_index] = 0

        if (
            pkv_index
            and pkv_index < len(args)  # pkv is in args
            and isinstance(args[pkv_index], (list, tuple))
            and isinstance(args[pkv_index][0], (list, tuple))
        ):
            if len(args[pkv_index][0]) == 2:
                args[pkv_index] = DynamicCache.from_legacy_cache(args[pkv_index])
            elif len(args[pkv_index][0]) == 4:
                args[pkv_index] = EncoderDecoderCache.from_legacy_cache(args[pkv_index])
            else:
                raise ValueError(
                    f"past_key_values should have either 2 or 4 elements, but it has {len(args[pkv_index][0])} elements"
                )
        elif (
            "past_key_values" in kwargs  # pkv is in kwargs
            and isinstance(kwargs["past_key_values"], (list, tuple))
            and isinstance(kwargs["past_key_values"][0], (list, tuple))
        ):
            if len(kwargs["past_key_values"][0]) == 2:
                kwargs["past_key_values"] = DynamicCache.from_legacy_cache(kwargs["past_key_values"])
            elif len(kwargs["past_key_values"][0]) == 4:
                kwargs["past_key_values"] = EncoderDecoderCache.from_legacy_cache(kwargs["past_key_values"])
            else:
                raise ValueError(
                    "past_key_values should have either 2 or 4 elements, "
                    f"but it has {len(kwargs['past_key_values'][0])} elements"
                )

        outputs = orig_forward(*args, **kwargs)

        if isinstance(outputs, dict) and isinstance(
            outputs.get("past_key_values"), (DynamicCache, EncoderDecoderCache)
        ):
            outputs["past_key_values"] = outputs["past_key_values"].to_legacy_cache()

        return outputs

    setattr(pytorch_model, orig_forward_name, patched_forward)
    logger.debug("PyTorch model patched for transformers v%s.", transformers.__version__)


@torch.no_grad()
def _export_pytorch_model(
    pytorch_model: torch.nn.Module,
    dummy_inputs,
    io_config,
    config: type[BasePassConfig],
    device: Union[str, torch.device],
    dynamo: bool,
    torch_dtype: Optional[torch.dtype] = None,
) -> ir.Model:
    """Export a torch.nn.Module to ONNX and return the loaded ONNX model.

    :param pytorch_model: the torch.nn.Module to export
    :param dummy_inputs: the dummy inputs to the model. Can be None if using dynamo_exporter
    :param io_config: the io_config for the model. This consists of the input and output names, and dynamic axes
    :param config: the config for the pass
    :param device: the device to use for conversion
    :param dynamo: whether to use the dynamo=True option for export
    :param torch_dtype: the dtype to cast the model to before conversion
    :param tempdir: directory to use for temporary files
    """
    from olive.common.hf.peft import make_export_compatible_peft
    from olive.common.hf.quant import make_export_compatible_quant

    device = torch.device(device)
    use_gpu = device != torch.device("cpu")
    if use_gpu and torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.debug("Converting model on device %s with dtype %s.", device, torch_dtype)
    pytorch_model.to(device)

    dummy_inputs = tensor_data_to_dtype(dummy_inputs, torch_dtype)
    dummy_inputs = tensor_data_to_device(dummy_inputs, device)

    if isinstance(pytorch_model, torch.jit.RecursiveScriptModule):
        pytorch_model = TraceModelWrapper(pytorch_model)
    pytorch_model = make_export_compatible_peft(pytorch_model, merge_weights=config.merge_adapter_weights)
    pytorch_model = make_export_compatible_quant(pytorch_model, config.use_dynamo_exporter)
    # cast to dtype, want all modules including lora layers and quant linears in the same dtype
    if torch_dtype:
        pytorch_model = pytorch_model.to(torch_dtype)

    # get input and output names, and dynamic axes
    assert io_config is not None, "Cannot get io_config for the model."
    io_config = validate_config(io_config, IoConfig)
    # If dynamic is False, set dynamic_axes and dynamic_shapes to None
    if not config.dynamic:
        io_config.dynamic_axes = None
        io_config.dynamic_shapes = None

    # Create tempdir for the case when fallback or dynamo=False is used. This is because when fallback
    # is taken, the old export always writes a model to the disk. When that happens we need to
    # load the model back into IR and load all the external tensor to memory
    with tempfile.TemporaryDirectory(prefix="olive_tmp") as tmp_dir:
        if dynamo:
            # Take the "release" version so that dev builds like 2.5.0dev1234 are treated as 2.5.0
            if _torch_is_older_than("2.7.0") and (
                io_config.dynamic_axes is not None or io_config.dynamic_shapes is not None
            ):
                logger.warning(
                    "We recommend PyTorch version 2.7.0 or later for dynamic_shapes support. "
                    "Please upgrade to PyTorch 2.7.0 or newer if you need dynamic shapes.",
                )
            # The new "dynamo" api is torch.onnx.export with dynamo=True
            if _torch_is_older_than("2.6.0"):
                raise RuntimeError(
                    f"torch.onnx.export(..., dynamo=True) is not available for torch version {torch.__version__}. "
                    "Please upgrade PyTorch to 2.6.0 or above."
                )

            if isinstance(dummy_inputs, dict):
                dummy_kwargs = dummy_inputs
                dummy_inputs = ()
            else:
                dummy_kwargs = {}
                dummy_inputs = tuple(dummy_inputs)

            # Apply patches for DynamicCache / past_key_values compatibility
            if version.parse(transformers.__version__) >= version.parse("5.0"):
                # transformers >= 5.0: DynamicCache refactored to use DynamicLayer

                _register_dynamic_cache_export_support()
                _patch_dynamic_layer_for_export()
                model_config = getattr(pytorch_model, "config", None)
                dummy_kwargs = _convert_past_key_values_to_dynamic_cache(dummy_kwargs, config=model_config)
                if io_config.dynamic_shapes:
                    io_config.dynamic_shapes = _convert_dynamic_shapes_for_dynamic_cache(io_config.dynamic_shapes)
            else:
                # transformers < 5.0: patch forward to convert list <-> DynamicCache
                _patch_model_if_necessary(pytorch_model)

            # NOTE: Usually validation is done in io_config.py, but because
            # dynamic_shapes has nested complexity, and it can't be validated multiple
            # times like others, we validate it here.
            io_config.dynamic_shapes, dummy_inputs, dummy_kwargs = _validate_dynamic_shapes(
                io_config.dynamic_shapes, dummy_inputs, dummy_kwargs, pytorch_model
            )
            # torch.export requires strict type match between inputs and dynamic_shapes;
            # _validate_dynamic_shapes may return OrderedDict, so convert back to plain dict
            if isinstance(io_config.dynamic_shapes, collections.OrderedDict):
                io_config.dynamic_shapes = dict(io_config.dynamic_shapes)
            if isinstance(dummy_kwargs, collections.OrderedDict):
                dummy_kwargs = dict(dummy_kwargs)

            # When dynamo=True, PyTorch prefers dynamic_shapes over dynamic_axes.
            # If dynamic_shapes is None and fallback is enabled, don't pass dynamic_axes
            # to avoid conversion errors. The fallback path will handle dynamic axes.
            dynamic_axes_for_export = io_config.dynamic_axes if io_config.dynamic_shapes else None

            onnx_program = torch.onnx.export(  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
                pytorch_model,
                dummy_inputs,
                kwargs=dummy_kwargs,
                opset_version=config.target_opset,
                input_names=io_config.input_names,
                output_names=io_config.output_names,
                dynamic_axes=dynamic_axes_for_export,
                dynamic_shapes=io_config.dynamic_shapes,
                dynamo=True,
                optimize=config.optimize,
                report=logger.isEnabledFor(logging.DEBUG),
            )
            assert onnx_program is not None
            model = onnx_program.model
            # We can run load_to_model on all models: If the model is from dynamo=True,
            # there is no external tensor so nothing will happen; if the model is from
            # fallback, the external tensors will be loaded into memory so the tempdir
            # can be removed.
            ir.external_data.load_to_model(model)
        else:
            dynamo_args = {}
            if not _torch_is_older_than("2.9.0"):
                # default is True in 2.9.0 and later
                dynamo_args["dynamo"] = False

            tmp_model_path = resolve_onnx_path(tmp_dir)

            torch.onnx.export(
                pytorch_model,
                dummy_inputs,
                tmp_model_path,
                export_params=True,
                opset_version=config.target_opset,
                input_names=io_config.input_names,
                output_names=io_config.output_names,
                dynamic_axes=io_config.dynamic_axes,
                **dynamo_args,
            )
            # After the line below, the model is loaded into memory, so it's safe to
            # delete previously exported file(s)
            # loading using onnx for now since ir.external_data.load_to_model doesn't load constants from external files
            # it's also faster this way
            model = ir.serde.deserialize_model(onnx.load(tmp_model_path))

            # Workaround as described under IoConfig.string_to_int_dim_params: change numeric dim_param to dim_value
            if io_config.string_to_int_dim_params:
                string_to_int_dim_params = set(io_config.string_to_int_dim_params)
                for output in model.graph.outputs:
                    new_shape = []
                    # Create a new shape only when any dimension was changed
                    changed = False
                    for dim in output.shape:
                        if isinstance(dim, ir.SymbolicDim) and dim.value in string_to_int_dim_params:
                            new_shape.append(int(dim.value))
                            changed = True
                        else:
                            new_shape.append(dim)
                    if changed:
                        output.shape = ir.Shape(new_shape)

    # Reset to CPU so the resource consumed on GPU could be free.
    if use_gpu:
        pytorch_model.to("cpu")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return model


def _get_dummy_inputs(
    model: Union[HfModelHandler, PyTorchModelHandler], config: type[BasePassConfig]
) -> Union[dict, Any]:
    """Get dummy inputs for the model."""
    return model.get_dummy_inputs(
        filter_hook=(model.merge_kv_cache_hook if config.use_dynamo_exporter else model.merge_kv_cache_to_tuple_hook),
        filter_hook_kwargs={
            "past_kv_names": config.past_key_value_name,
        },
    )


def _export_ranked_model(params):
    """Export one rank of a DistributedHfModel to ONNX and save the model to the output path.

    :param params: a tuple of (pass_config, model_config, world_size, device, local_rank, output_dirpath)
        pass_config: the config for the pass
        model_config: the config for the DistributedHfModel
        device: the device to use for conversion
        torch_dtype: the dtype to cast the model to before conversion
        local_rank: the rank of the current process as well as the rank of the model to be converted
        output_dirpath: the path to the directory to save the model. The .onnx model will be saved in this
            directory with the name specified by DistributedOnnxModel.DEFAULT_RANKED_MODEL_NAME_FORMAT
    """
    pass_config, model_config, device, torch_dtype, local_rank, output_dirpath = params

    model_type = model_config.get("model_attributes", {}).get("model_type")

    if model_type == "llama":
        from olive.passes.pytorch.tensor_parallel_llama2 import (
            replace_llama2_tensor_parallel_layers as replace_tensor_parallel_layers,
        )
        from olive.passes.pytorch.tensor_parallel_llama2 import (
            restore_llama2_tensor_parallel_layers as restore_tensor_parallel_layers,
        )
    else:
        raise ValueError(f"Unsupported model type '{model_type}' for conversion pass")

    output_filename = DistributedOnnxModelHandler.DEFAULT_RANKED_MODEL_NAME_FORMAT.format(local_rank)
    output_filepath = resolve_onnx_path(output_dirpath, output_filename)

    restore_args = replace_tensor_parallel_layers()
    try:
        input_model = DistributedHfModelHandler(**model_config)

        olive_pytorch_model = input_model.load_model(local_rank)
        dummy_inputs = _get_dummy_inputs(olive_pytorch_model, pass_config)
        io_config = None if pass_config.use_dynamo_exporter else olive_pytorch_model.io_config
        pytorch_model = olive_pytorch_model.prepare_session(rank=local_rank)

        ranked_onnx_model = _export_pytorch_model(
            pytorch_model,
            dummy_inputs,
            io_config,
            pass_config,
            device,
            dynamo=False,
            torch_dtype=torch_dtype,
        )

        # save the model to the output path
        ir_model_to_olive_model(ranked_onnx_model, output_filepath, pass_config)
    finally:
        restore_tensor_parallel_layers(restore_args)

    return 1  # Return 1 for success.


def _prepare_hf_model(model: HfModelHandler, device: str, torch_dtype: Optional[torch.dtype] = None) -> HfModelHandler:
    """Prepare the HfModelHandler for conversion.

    This method handles the following cases:
    1. HfModelHandler with no load kwargs
        - no need to change the model
    2. HfModelHandler with load kwargs
        - update load_kwargs.torch_dtype if torch_dtype is specified
        - if torch_dtype not specified, make sure the load kwargs specify a dtype that is supported for
            conversion on the specified device
        - if quantization_method == "bitsandbytes" and load_in_4bit is True
            - remove quantization config from the load kwargs
            - find quantized modules and add them to the model attributes
            - the onnx model must be quantized using OnnxBnb4Quantization pass after conversion
    """
    from olive.common.hf.peft import is_peft_model

    if not model.load_kwargs:
        return model

    model_attributes = deepcopy(model.model_attributes or {})
    load_kwargs = model.load_kwargs
    model_dtype = load_kwargs.get_torch_dtype()
    new_load_kwargs = deepcopy(load_kwargs.model_dump())

    if torch_dtype and torch_dtype != model_dtype:
        # if the load kwargs specify a different dtype, update the load kwargs
        logger.debug(
            "Changing torch_dtype in load kwargs from %s to %s.",
            load_kwargs.get_torch_dtype(),
            torch_dtype,
        )
        new_load_kwargs["torch_dtype"] = torch_dtype
        model_attributes["torch_dtype"] = str(torch_dtype).replace("torch.", "")

    if load_kwargs.quantization_method == "bitsandbytes" and load_kwargs.quantization_config["load_in_4bit"]:
        logger.debug(
            "Bitsandbytes 4bit quantization is not supported for conversion. The quantization config is removed"
            " from the load kwargs. Use OnnxBnb4Quantization pass after conversion to quantize the"
            " model."
        )
        new_load_kwargs["quantization_method"] = None
        new_load_kwargs["quantization_config"] = None
        model_attributes["quantization_config"] = load_kwargs.quantization_config
        if "quantized_modules" not in model_attributes:
            # find and add quantized modules to the model attributes
            # the QLoRA pass already adds quantized_modules to the model attributes, so this will not be
            # executed if the model was generated by QLoRA
            quantized_model = model.load_model(cache_model=False)

            # if PeftModel, need to unload adapter before finding quantized modules
            if is_peft_model(quantized_model):
                quantized_model = quantized_model.unload()

            import bitsandbytes as bnb

            model_attributes["quantized_modules"] = find_submodules(quantized_model, bnb.nn.Linear4bit)

    model_config = model.to_json()["config"]
    model_config["load_kwargs"] = new_load_kwargs
    model_config["model_attributes"] = model_attributes
    return HfModelHandler(**model_config)


class OnnxConversion(Pass):
    """Convert a PyTorch model to ONNX model using torch.onnx.export on CPU."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            **get_user_script_data_config(),
            **get_external_data_config(),
            "target_opset": PassConfigParam(
                type_=int, default_value=20, description="The version of the default (ai.onnx) opset to target."
            ),
            "use_dynamo_exporter": PassConfigParam(
                type_=bool, default_value=False, description="Whether to use dynamo_export API to export ONNX model."
            ),
            "past_key_value_name": PassConfigParam(
                type_=str,
                default_value="past_key_values",
                description=(
                    "The arguments name to point to past key values. For model loaded from huggingface, "
                    "it is 'past_key_values'. Basically, it is used only when `use_dynamo_exporter` is True."
                ),
            ),
            "device": PassConfigParam(
                type_=str,
                description=(
                    "The device to use for conversion, e.g., 'cuda' or 'cpu'. If not specified, will use 'cpu' for"
                    " PyTorch model and 'cuda' for DistributedHfModel."
                ),
            ),
            "torch_dtype": PassConfigParam(
                type_=str,
                description=(
                    "The dtype to cast the model to before conversion, e.g., 'float32' or 'float16'. If not specified,"
                    " will use the model as is."
                ),
            ),
            "parallel_jobs": PassConfigParam(
                type_=int,
                default=multiprocessing.cpu_count(),
                required=False,
                description="Number of parallel jobs. Defaulted to number of CPUs. Set it to 0 to disable.",
            ),
            "merge_adapter_weights": PassConfigParam(
                type_=bool,
                default_value=False,
                description=(
                    "Whether to merge adapter weights before conversion. "
                    "After merging, the model structure is consistent with base model. "
                    "That is useful if you cannot run conversion for some fine-tuned "
                    "models with adapter weights"
                ),
            ),
            "save_metadata_for_token_generation": PassConfigParam(
                type_=bool,
                default_value=False,
                description=(
                    "Whether to save metadata for token generation or not. "
                    "Includes config.json, generation_config.json, and tokenizer related files."
                ),
            ),
            "optimize": PassConfigParam(
                type_=bool,
                default_value=True,
                description="Whether to export the model with constant folding and redundancies elimination.",
            ),
            "dynamic": PassConfigParam(
                type_=bool, default_value=True, description="Whether to export the model with dynamic axes/shapes."
            ),
        }

    def _run_for_config(
        self,
        model: Union[DistributedHfModelHandler, HfModelHandler, PyTorchModelHandler],
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> Union[DistributedOnnxModelHandler, ONNXModelHandler]:
        output_model = self._run_for_config_internal(model, config, output_model_path)

        if isinstance(model, HfModelHandler):
            output_model.model_attributes = model_attributes = output_model.model_attributes or {}
            model_attributes["hf_task"] = model.task
            model_attributes["type"] = model.model_type

            if config.save_metadata_for_token_generation:
                # output_model can only be an ONNXModelHandler
                output_dir = output_model.change_model_path_to_dir()
                model_attributes["additional_files"] = additional_files = model_attributes.get("additional_files", [])
                # quantization config is already popped from the model and included in model_attributes
                # don't want the information to be saved in metadata (issues with generation config save)
                additional_files.extend(model.save_metadata(str(output_dir), exclude_load_keys=["quantization_config"]))

        return output_model

    def _run_for_config_internal(
        self,
        model: Union[DiffusersModelHandler, DistributedHfModelHandler, HfModelHandler, PyTorchModelHandler],
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> Union[CompositeModelHandler, DistributedOnnxModelHandler, ONNXModelHandler]:
        # get the device to use for conversion
        # default to "cpu" for PyTorchModelHandler and "cuda" for DistributedHfModel
        device = config.device or "cpu"
        # get the dtype to use for conversion
        torch_dtype = resolve_torch_dtype(config.torch_dtype) if config.torch_dtype else None
        if torch_dtype == torch.float16 and device == "cpu":
            logger.debug(
                "Converting model to float16 on CPU. This might fail for some models. If the conversion fails or model"
                " is incorrect, try converting the model on GPU or convert in float32 and use"
                " OrtTransformerOptimization/OnnxFloatToFloat16 pass after this pass."
            )

        if isinstance(model, DiffusersModelHandler):
            return self._convert_diffusers_model(model, config, output_model_path, device, torch_dtype)

        if isinstance(model, DistributedHfModelHandler):
            if not config.device:
                device = "cuda"
            return self._convert_distributed_model_on_device(model, config, output_model_path, device, torch_dtype)

        return self._convert_model_on_device(model, config, output_model_path, device, torch_dtype)

    def _convert_model_on_device(
        self,
        model: Union[HfModelHandler, PyTorchModelHandler],
        config: type[BasePassConfig],
        output_model_path: str,
        device: str,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> ONNXModelHandler:
        """Convert an HfModelHandler or PyTorchModelHandler to an ONNXModelHandler."""
        # prepare the model for conversion
        if isinstance(model, HfModelHandler):
            # optimum export config needs the loaded model to get io_config so we create a new model handler
            # which will be used to load the model and get the io_config
            model = _prepare_hf_model(model, device, torch_dtype)

        # load the model
        pytorch_model = model.load_model(cache_model=False)
        pytorch_model.eval()

        # get dummy inputs
        dummy_inputs = _get_dummy_inputs(model, config)
        io_config = model.io_config

        model_attributes = deepcopy(model.model_attributes or {})

        # add split information if present
        split_assignments = model_attributes.get("split_assignments")
        if split_assignments:
            split_assignment_encoded = ";".join([f"{k}={v}" for k, v in split_assignments.items()])
        else:
            split_assignment_encoded = None

        output_model_path = resolve_onnx_path(output_model_path)
        ir_model = _export_pytorch_model(
            pytorch_model,
            dummy_inputs,
            io_config=io_config,
            config=config,
            device=device,
            dynamo=config.use_dynamo_exporter,
            torch_dtype=torch_dtype,
        )
        if split_assignment_encoded:
            ir_model.metadata_props["split_assignments"] = split_assignment_encoded
        output_model = ir_model_to_olive_model(ir_model, output_model_path, config)

        output_model.model_attributes = model_attributes
        return output_model

    def _convert_diffusers_model(
        self,
        model: DiffusersModelHandler,
        config: type[BasePassConfig],
        output_model_path: str,
        device: str,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> CompositeModelHandler:
        """Convert a DiffusersModelHandler to ONNX models.

        Args:
            model: The DiffusersModelHandler to convert.
            config: The pass configuration.
            output_model_path: The output path for the ONNX models.
            device: The device to use for conversion.
            torch_dtype: The dtype to cast the model to before conversion.

        Returns:
            CompositeModelHandler containing ONNXModelHandler for each component.

        """
        from olive.common.hf.io_config import generate_diffusers_dummy_inputs, get_diffusers_io_config
        from olive.common.hf.peft import make_export_compatible_peft

        output_dir = Path(output_model_path).with_suffix("")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load the pipeline (this also loads LoRA if adapter_path is set)
        pipeline = model.load_model(cache_model=False)

        # Get the pipeline type and exportable components
        pipeline_type = model.get_pipeline_type()
        exportable_components = model.get_exportable_components()

        logger.info("Exporting diffusers model: %s (type: %s)", model.model_path, pipeline_type)

        component_handlers = []
        component_names = []

        for component_name in exportable_components:
            # Get the component model and config
            component_model, component_config = self._get_diffusers_component(pipeline, component_name)

            if component_model is None:
                logger.warning("Component %s not found in pipeline, skipping", component_name)
                continue

            # Apply LoRA compatibility for unet/transformer if adapter was loaded
            if model.adapter_path and component_name in (DiffusersComponent.UNET, DiffusersComponent.TRANSFORMER):
                component_model = make_export_compatible_peft(
                    component_model,
                    merge_weights=config.merge_adapter_weights,
                )

            # Move to device and dtype
            component_model = component_model.to(device)
            if torch_dtype:
                component_model = component_model.to(torch_dtype)
            component_model.eval()

            # Generate dummy inputs using new task-driven API
            dummy_inputs = generate_diffusers_dummy_inputs(
                component_name=component_name,
                config=component_config,
                pipeline=pipeline_type,
            )

            # Get IO config using new task-driven API
            io_config = get_diffusers_io_config(
                component_name=component_name,
                config=component_config,
                pipeline=pipeline_type,
            )

            # Create output directory for this component
            component_dir = output_dir / component_name
            component_dir.mkdir(parents=True, exist_ok=True)
            component_output_path = str(component_dir / "model.onnx")

            # Export using _export_pytorch_model
            ir_model = _export_pytorch_model(
                component_model,
                dummy_inputs,
                io_config=io_config,
                config=config,
                device=device,
                dynamo=True,
                torch_dtype=torch_dtype,
            )

            # Save the model
            output_model = ir_model_to_olive_model(ir_model, component_output_path, config)
            component_handlers.append(output_model)
            component_names.append(component_name)
            logger.info("Exported %s to %s", component_name, component_output_path)

            # Clean up GPU memory
            del component_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not component_handlers:
            raise ValueError(
                f"No ONNX components exported from '{model.model_path}'. "
                "Ensure the model is a valid diffusion pipeline."
            )

        model_attributes = deepcopy(model.model_attributes or {})
        model_attributes["model_variant"] = str(model.detected_model_variant)
        model_attributes["no_flatten"] = True

        return CompositeModelHandler(
            model_components=component_handlers,
            model_component_names=component_names,
            model_path=str(output_dir),
            model_attributes=model_attributes,
        )

    def _get_diffusers_component(
        self, pipeline, component_name: str
    ) -> tuple[Optional[torch.nn.Module], Optional[Any]]:
        """Get a component model and its config from the pipeline.

        Args:
            pipeline: The diffusion pipeline.
            component_name: Name of the component to get.

        Returns:
            Tuple of (component_model, component_config), or (None, None) if not found.

        """
        from olive.model.utils.diffusers_utils import get_vae_decoder, get_vae_encoder

        # Handle VAE encoder/decoder specially
        if component_name == DiffusersComponent.VAE_ENCODER:
            vae = getattr(pipeline, "vae", None)
            if vae is None:
                return None, None
            return get_vae_encoder(vae), vae.config

        if component_name == DiffusersComponent.VAE_DECODER:
            vae = getattr(pipeline, "vae", None)
            if vae is None:
                return None, None
            return get_vae_decoder(vae), vae.config

        # For other components, get directly from pipeline
        component = getattr(pipeline, component_name, None)
        if component is None:
            return None, None

        config = getattr(component, "config", None)
        return component, config

    def _convert_distributed_model_on_device(
        self,
        model: DistributedHfModelHandler,
        config: type[BasePassConfig],
        output_model_path: str,
        device: str,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> DistributedOnnxModelHandler:
        """Convert a DistributedHfModel to a DistributedOnnxModel."""
        pass_config = config
        model_config = model.to_json()["config"]
        world_size = model.num_ranks
        output_model_path = str(Path(output_model_path).with_suffix(""))
        use_gpu = torch.device(device) != torch.device("cpu")

        params = [
            (
                pass_config,
                model_config,
                torch.device("cuda", rank) if use_gpu else torch.device("cpu"),
                torch_dtype,
                rank,
                output_model_path,
            )
            for rank in range(world_size)
        ]

        max_parallel_jobs = min(world_size, config.parallel_jobs or multiprocessing.cpu_count())
        if max_parallel_jobs <= 1:
            results = [_export_ranked_model(_) for _ in params]
        else:
            context = multiprocessing.get_context("spawn")
            with context.Pool(processes=max_parallel_jobs) as pool:
                results = pool.map(_export_ranked_model, params)

        if world_size != sum(results):
            raise RuntimeError("Failed to convert models")

        return DistributedOnnxModelHandler(
            model_path=output_model_path,
            model_name_pattern=DistributedOnnxModelHandler.DEFAULT_RANKED_MODEL_NAME_FORMAT,
            num_ranks=world_size,
            model_attributes=model.model_attributes,
        )


class OnnxOpVersionConversion(Pass):
    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        latest_opset_version = onnx.defs.onnx_opset_version()

        config = {
            "target_opset": PassConfigParam(
                type_=int,
                default_value=latest_opset_version,
                description="The version of the default (ai.onnx) opset to target. Default: latest opset version.",
            ),
        }
        config.update(get_external_data_config())
        return config

    def _run_for_config(
        self, model: ONNXModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> ONNXModelHandler:
        output_model_path = resolve_onnx_path(output_model_path)
        model_ir = model.load_ir_model()
        version_converter.convert_version(model_ir, config.target_opset, fallback=True)
        return ir_model_to_olive_model(model_ir, output_model_path, config)


def _validate_dynamic_shapes(dynamic_shapes, dummy_inputs, dummy_kwargs, model):
    """Validate dynamic_shapes.

    This function validates two things:

    (1) To have a valid format of dynamic_shapes, we need to make sure the axes are converted to int.
        It was string in the JSON format.
    (2) To make sure the dynamic_shapes is in the same tree structure as dummy_inputs.

    :param dynamic_shapes: the dynamic_shapes to validate
    :param dummy_inputs: the dummy_inputs to align the dynamic_shapes format

    :return: the validated dynamic_shapes
    """
    if not dynamic_shapes:
        return dynamic_shapes, dummy_inputs, dummy_kwargs

    from torch.utils import _pytree

    # Align tree spec only for not transformers.Cache.
    if len(dummy_inputs) == 0:
        for k, v in dummy_kwargs.items():
            if not isinstance(v, transformers.Cache) and k in dynamic_shapes:
                input_tree_spec = _pytree.tree_flatten(v)[1]
                flatten_dynamic_shapes = get_the_flattened_and_tree_spec(dynamic_shapes[k], leaf_is_str=False)[0]
                dynamic_shapes[k] = _pytree.tree_unflatten(flatten_dynamic_shapes, input_tree_spec)
    else:
        for i, v in enumerate(dummy_inputs):
            if not isinstance(v, transformers.Cache):
                input_tree_spec = _pytree.tree_flatten(v)[1]
                flatten_dynamic_shapes = get_the_flattened_and_tree_spec(dynamic_shapes[i], leaf_is_str=False)[0]
                dynamic_shapes[i] = _pytree.tree_unflatten(flatten_dynamic_shapes, input_tree_spec)

    # The input can only be either args or kwargs according to line 237.
    if len(dummy_inputs) == 0:
        # NOTE: dynamic_shapes need to follow the same model.forward signature when it's referring to kwargs.
        param_order = list(inspect.signature(model.forward).parameters)
        # Sort io_config.dynamic_shapes based on this order
        dynamic_shapes = collections.OrderedDict(
            sorted(dynamic_shapes.items(), key=lambda item: param_order.index(item[0]))
        )
        dummy_kwargs = collections.OrderedDict(
            sorted(dummy_kwargs.items(), key=lambda item: param_order.index(item[0]))
        )
        return dynamic_shapes, dummy_inputs, dummy_kwargs
    # If dynamic_shapes and dummy_inputs are both list/tuple, we don't need to sort.
    # dummy_inputs is args
    return dynamic_shapes, dummy_inputs, dummy_kwargs

from typing import Optional

from .options import DebugOptions
from .ortmodule import ORTModule
from . import _utils


def prepare_model_for_parallel_pipeline(model, debug_options: Optional[DebugOptions] = None) -> None:
    """
    Prepares a given model for parallel pipeline execution by wrapping its submodules
    in ORTModule based on a device map.

    Parameters:
    - model: The model to prepare. Must have a `hf_device_map` attribute as this implementation only supports huggingface accelerate parallel pipeline.
    - debug_options: Optional DebugOptions instance to customize ORTModule behavior. If None,
                     no additional debug options are set.

    Raises:
    - ValueError: If the model does not have a `hf_device_map` attribute.
    """

    # Check if model is dispatched to multiple devices
    if not _utils.is_model_dispatched(model):
        raise ValueError("The model is not dispatched to multiple devices, use ORTModule to wrap your model.")

    if not hasattr(model, "hf_device_map"):
        raise ValueError(
            "The provided model does not have an 'hf_device_map' attribute, which is "
            "required for ORTModule parallel pipeline support. "
            "Please ensure the model is compatible and properly configured."
        )

    hf_device_map = model.hf_device_map

    for name, device in hf_device_map.items():
        # Retrieve submodule using a safe navigation method
        layer = model
        for part in name.split("."):
            layer = getattr(layer, part)

        # Construct ORTModule with debug options if provided
        if debug_options:
            new_onnx_prefix = str(device).replace(":", "_") + "_" + debug_options.onnx_prefix

            parallel_debug_options = DebugOptions(
                debug_options.log_level, debug_options.save_onnx, new_onnx_prefix, debug_options.save_path
            )
            setattr(model, name, ORTModule(layer, parallel_debug_options))
        else:
            setattr(model, name, ORTModule(layer))

from typing import Optional
import os

from .options import DebugOptions
from .ortmodule import ORTModule
from . import _utils


def wrap_submodule_with_ort(model, submodule_path: str, debug_options=None):
    """
    Wraps a specific submodule within a model with ORTModule and applies debug options.
    This function assumes a direct path to the submodule for replacement.
    """

    path_parts = submodule_path.split(".")
    parent_module = model
    for part in path_parts[:-1]:
        parent_module = getattr(parent_module, part)

    target_name_or_index = path_parts[-1]

    # Wrap the target submodule with ORTModule and DebugOptions
    try:
        target_index = int(target_name_or_index)
        is_index = True
    except ValueError:
        is_index = False

    if is_index:
        parent_module[target_index] = ORTModule(parent_module[target_index], debug_options)
    else:
        setattr(
            parent_module, target_name_or_index, ORTModule(getattr(parent_module, target_name_or_index), debug_options)
        )


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

    for submodule_path, device in hf_device_map.items():
        # Construct ORTModule with debug options if provided
        if debug_options:
            new_onnx_prefix = str(device).replace(":", "_") + "_" + debug_options.onnx_prefix
            device_folder_name = "device_" + str(device).replace(":", "_")
            device_save_path = os.path.join(debug_options.save_path, device_folder_name)
            if not os.path.exists(device_save_path):
                os.makedirs(device_save_path)

            parallel_debug_options = DebugOptions(
                debug_options.log_level, debug_options.save_onnx, new_onnx_prefix, device_save_path
            )
            wrap_submodule_with_ort(model, submodule_path, parallel_debug_options)
        else:
            wrap_submodule_with_ort(model, submodule_path)

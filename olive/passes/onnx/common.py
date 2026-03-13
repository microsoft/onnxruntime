# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Optional, Union

import onnx
from onnx import external_data_helper
from onnxscript import ir
from onnxscript.optimizer._constant_folding import FOLDED_FROM_KEY

from olive.common.utils import StrEnumBase, hardlink_copy_file
from olive.model import CompositeModelHandler, ONNXModelHandler
from olive.passes.onnx.onnx_dag import OnnxDAG
from olive.passes.pass_config import BasePassConfig, PassConfigParam
from olive.resource_path import LocalFile, LocalFolder

logger = logging.getLogger(__name__)

_LARGE_IR_MODEL_THRESHOLD = 1536 * 1024 * 1024  # 1536MB


class AdapterType(StrEnumBase):
    LORA = "lora"
    DORA = "dora"
    LOHA = "loha"


def get_external_data_config() -> dict[str, PassConfigParam]:
    return {
        "save_as_external_data": PassConfigParam(
            type_=bool,
            default_value=False,
            description=(
                "Serializes tensor data to separate files instead of directly in the ONNX file. Large models (>2GB)"
                " may be forced to save external data regardless of the value of this parameter."
            ),
        ),
        "all_tensors_to_one_file": PassConfigParam(
            type_=bool,
            default_value=True,
            description=(
                "Effective only if save_as_external_data is True. If true, save all tensors to one external file"
                " specified by 'external_data_name'. If false, save each tensor to a file named with the tensor name."
            ),
        ),
        "external_data_name": PassConfigParam(
            type_=str,
            default_value=None,
            description=(
                "Effective only if all_tensors_to_one_file is True and save_as_external_data is True. If not specified,"
                " the external data file will be named with <model_path_name>.data"
            ),
        ),
        "size_threshold": PassConfigParam(
            type_=int,
            default_value=1024,
            description=(
                "Effective only if save_as_external_data is True. Threshold for size of data. Only when tensor's data"
                " is >= the size_threshold it will be converted to external data. To convert every tensor with raw data"
                " to external data set size_threshold=0."
            ),
        ),
        "convert_attribute": PassConfigParam(
            type_=bool,
            default_value=False,
            description=(
                "Effective only if save_as_external_data is True. If true, convert all tensors to external data If"
                " false, convert only non-attribute tensors to external data"
            ),
        ),
    }


def add_version_metadata_to_model_proto(model: onnx.ModelProto) -> onnx.ModelProto:
    olive_version = None
    try:
        import olive

        olive_version = getattr(olive, "__version__", "unknown")
    except Exception:
        olive_version = "unknown"

    for md in model.metadata_props:
        if md.key == "olive_version":
            md.value = olive_version
            return model

    md = model.metadata_props.add()
    md.key = "olive_version"
    md.value = olive_version

    return model


def model_proto_to_file(
    model: onnx.ModelProto,
    output_path: Union[str, Path],
    save_as_external_data: Optional[bool] = False,
    all_tensors_to_one_file: Optional[bool] = True,
    external_data_name: Optional[Union[str, Path]] = None,
    size_threshold: Optional[int] = 1024,
    convert_attribute: Optional[bool] = False,
) -> bool:
    """Save the ONNX model to the specified path.

    :param model: The ONNX model to save.
    :param output_path: The path to save the ONNX model to.
    :param save_as_external_data: If True, save tensor data to separate files instead of directly in the ONNX file.
        Large models (>2GB) may be forced to save external data regardless of the value of this parameter.
    :param all_tensors_to_one_file: Effective only if save_as_external_data is True. If True, save all tensors to one
        external file specified by 'external_data_name'. If False, save each tensor to a file named with the tensor
        name.
    :param external_data_name: Effective only if all_tensors_to_one_file is True and save_as_external_data is True.
        If not specified, the external data file will be named with <model_path_name>.data

    :return: True if the model has external data, False otherwise.
    """
    output_path = Path(output_path)
    if output_path.exists():
        logger.debug("Deleting existing onnx file: %s", output_path)
        output_path.unlink()

    # parent directory of .onnx file
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # model size probing may fail for very large models/external data. Only probe when needed.
    if not save_as_external_data:
        try:
            model_size = model.ByteSize()
        except Exception as e:
            logger.warning(
                "Failed to compute model size with ByteSize (%s). Saving tensors as external data.",
                e,
            )
            save_as_external_data = True
        else:
            # model size for large models might be negative (overflow?) on Windows
            # see https://github.com/onnx/onnx/issues/5861
            if model_size <= 0 or model_size >= onnx.checker.MAXIMUM_PROTOBUF:
                save_as_external_data = True
                logger.debug(
                    "Model is too large to save as a single file but 'save_as_external_data' is False. Saving"
                    " tensors as external data, regardless."
                )

    if not save_as_external_data:
        # Add olive version to metadata
        add_version_metadata_to_model_proto(model)
        # save model
        onnx.save_model(model, str(output_path))
        return False

    # location for external data
    external_data_path = output_dir / (external_data_name if external_data_name else f"{output_path.name}.data")
    location = external_data_path.name if all_tensors_to_one_file else None

    if all_tensors_to_one_file:
        if external_data_path.exists():
            # Delete the external data file. Otherwise, data will be appended to existing file.
            logger.info("Deleting existing external data file: %s", external_data_path)
            external_data_path.unlink()
    else:
        if any(output_dir.iterdir()):
            raise RuntimeError(f"Output directory ({output_dir}) for external data is not empty.")

    # Add olive version to metadata
    add_version_metadata_to_model_proto(model)
    # save model
    onnx.save_model(
        model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=all_tensors_to_one_file,
        location=location,
        size_threshold=size_threshold,
        convert_attribute=convert_attribute,
    )
    return True


def _get_external_data_name(output_path: Path, external_data_name: Optional[str]) -> str:
    return external_data_name if external_data_name else f"{output_path.name}.data"


def model_proto_to_olive_model(
    model_proto: onnx.ModelProto,
    output_model_path: Union[str, Path],
    external_data_config: Union[dict[str, Any], type[BasePassConfig]],
    check_model: bool = False,
    external_initializers_file_name: Optional[str] = None,
    constant_inputs_file_name: Optional[str] = None,
    force_model_dir: bool = False,
) -> ONNXModelHandler:
    """Save the ONNX model to the specified path and return the ONNXModelHandler.

    :param model_proto: The ONNX model to save.
    :param output_model_path: The path to save the ONNX model to.
    :param external_data_config: The external data configuration. Must be a dictionary with keys
        "save_as_external_data", "all_tensors_to_one_file", and "external_data_name".
    :param check_model: If True, run onnx.checker.check_model on the model before returning.
    :param external_initializers_file_name: The name of the external initializers file.
    :param constant_inputs_file_name: The name of the constant inputs file.
    :param force_model_dir: If True, use the parent directory of the output model path as the model directory
        regardless of whether external data is used.

    :return: The ONNXModelHandler.
    """
    config_keys = [
        "save_as_external_data",
        "all_tensors_to_one_file",
        "external_data_name",
        "size_threshold",
        "convert_attribute",
    ]
    if not isinstance(external_data_config, dict):
        external_data_config = external_data_config.model_dump()
    has_external_data = model_proto_to_file(
        model_proto, output_model_path, **{k: external_data_config[k] for k in config_keys if k in external_data_config}
    )
    if has_external_data or external_initializers_file_name or constant_inputs_file_name or force_model_dir:
        model_path = LocalFolder({"path": Path(output_model_path).parent})
        onnx_file_name = Path(output_model_path).name
    else:
        model_path = LocalFile({"path": output_model_path})
        onnx_file_name = None

    olive_model = ONNXModelHandler(
        model_path=model_path,
        onnx_file_name=onnx_file_name,
        external_initializers_file_name=external_initializers_file_name,
        constant_inputs_file_name=constant_inputs_file_name,
    )

    if check_model:
        onnx.checker.check_model(olive_model.model_path)

    return olive_model


def _count_initializer_size(graph: ir.Graph) -> int:
    """Count the total size of the initializers in bytes."""
    return sum(v.const_value.nbytes for v in graph.initializers.values() if v.const_value is not None)


def ir_model_to_olive_model(
    model: ir.Model,
    output_model_path: Union[str, Path],
    external_data_config: Union[dict[str, Any], type[BasePassConfig]],
) -> ONNXModelHandler:
    """Save the ONNX model to the specified path and return the ONNXModelHandler.

    When ``save_as_external_data`` in external_data_config is True:

    - If external_data_name is specified, external data will take this name; if
      not specified, the external data file will be named with <model_path_name>.data

    :param model: The ONNX IR model to save.
    :param output_model_path: The path to save the ONNX model to.
    :param external_data_config: The external data configuration. Must be a dictionary with keys
        "save_as_external_data", "external_data_name".

    :return: The ONNXModelHandler.
    """
    if not isinstance(external_data_config, dict):
        external_data_config = external_data_config.model_dump()

    save_as_external_data = external_data_config.get("save_as_external_data")
    # Save as external data if requested or if the model is large
    # Since we do not have a true estimate of the model architecture size for IR Model,
    # we count the size of all initializers and limit that to 1.5GB.
    initializer_size = _count_initializer_size(model.graph)
    is_large_model = initializer_size > _LARGE_IR_MODEL_THRESHOLD
    if is_large_model:
        logger.debug("Model is large (%s), saving as external data", initializer_size)
    save_as_external_data = save_as_external_data or is_large_model

    if save_as_external_data:
        external_data_name = _get_external_data_name(
            Path(output_model_path), external_data_config.get("external_data_name")
        )
        ir.save(model, output_model_path, external_data=external_data_name)

        logger.debug("Model was saved with external data: %s", external_data_name)
        model_path = LocalFolder({"path": Path(output_model_path).parent})
        onnx_file_name = Path(output_model_path).name
    else:
        ir.save(model, output_model_path)

        logger.debug("Model was not saved with external data")
        model_path = LocalFile({"path": output_model_path})
        onnx_file_name = None

    return ONNXModelHandler(model_path=model_path, onnx_file_name=onnx_file_name)


def get_external_data_file_names(model_path: Union[str, Path]) -> list[str]:
    """Get the external data file names from the model.

    :param model_path: Path to the model file.
    :return: List of external data file names.
    """
    file_names = set()
    for tensor in external_data_helper._get_all_tensors(  # pylint: disable=W0212
        onnx.load(model_path, load_external_data=False)
    ):
        if external_data_helper.uses_external_data(tensor):
            file_names.add(external_data_helper.ExternalDataInfo(tensor).location)
    return list(file_names)


def change_external_data_location(model_proto: onnx.ModelProto, new_location: str):
    """Change the external data location in the model.

    :param model_proto: The model proto to modify.
    :param new_location: The new location for the external data.
    """
    for tensor in external_data_helper._get_all_tensors(model_proto):  # pylint: disable=W0212
        if external_data_helper.uses_external_data(tensor):
            # set dummy raw_data since set_external_data expected the field
            tensor.raw_data = b""
            info = external_data_helper.ExternalDataInfo(tensor)
            external_data_helper.set_external_data(
                tensor, new_location, offset=info.offset, length=info.length, checksum=info.checksum
            )
            # clear the raw_data field to avoid saving it and overwriting the info above
            tensor.ClearField("raw_data")


def get_context_bin_file_names(model_path: Union[str, Path]) -> list[str]:
    """Get the context binary file names from the model.

    :param model_path: Path to the model file.
    :return: List of context binary file names.
    """
    file_names = set()
    for node in onnx.load(model_path, load_external_data=False).graph.node:
        if node.op_type == "EPContext":
            for attr in node.attribute:
                if attr.name == "ep_cache_context":
                    try:
                        file_names.add(attr.s.decode("utf-8"))
                    except UnicodeDecodeError:
                        # embedded context binary file
                        continue
    return list(file_names)


def copy_context_bin_files(
    model_path: Union[str, Path],
    model_dir: Union[str, Path],
    saved_cb_files: Optional[dict[str, str]] = None,
) -> bool:
    """Copy the context binary files to the model directory.

    :param model_path: Path to the original model file.
    :param model_dir: Directory to save the copied context binary files.
    :param saved_cb_files: A dictionary of original file paths to new file names for context binary files.
    :return: True if the model has context binary files, False otherwise.
    """
    saved_cb_files = {} if saved_cb_files is None else saved_cb_files

    model_path = Path(model_path).resolve()
    model_dir = Path(model_dir).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    # TODO(jambayk): consider renaming cb files
    cb_file_names = get_context_bin_file_names(model_path)
    for cb_file_name in cb_file_names:
        cb_file_path = str(model_path.parent / cb_file_name)
        dest_file_path = model_dir / cb_file_name

        if cb_file_path in saved_cb_files:
            continue
        elif dest_file_path.exists():
            # File already exists in destination, skip copying
            logger.info("Context binary file %s already exists in %s, skipping copy", cb_file_name, model_dir)
            saved_cb_files[cb_file_path] = cb_file_name
            continue
        elif cb_file_name in saved_cb_files.values():
            raise RuntimeError(
                f"Context binary file name {cb_file_name} already exists in {model_dir}. Please rename the file."
            )

        hardlink_copy_file(cb_file_path, dest_file_path)
        saved_cb_files[cb_file_path] = cb_file_name

    return bool(cb_file_names)


def resave_model(
    model_path: Union[str, Path],
    new_model_path: Union[str, Path],
    force_external_data: bool = False,
    saved_external_files: Optional[dict[str, str]] = None,
) -> bool:
    """Resave the model along with external data files.

    :param model_path: Path to the original model file.
    :param new_model_path: Path to the new model file.
    :param force_external_data: If True, force the model to be saved with external data.
    :param saved_external_files: A dictionary of original file paths to new file names for external data files.
        Reuse the same file name if the the original file path is already in the dictionary.
        Else, the new file name will be <new_model_path>.data and this dictionary will be updated with the new
        file name.
    :return: True if the model has external data, False otherwise.
    """
    saved_external_files = {} if saved_external_files is None else saved_external_files

    model_path = Path(model_path).resolve()
    new_model_path = Path(new_model_path).resolve()
    assert new_model_path.suffix == ".onnx", "new_model_path must be .onnx file"
    new_model_path.parent.mkdir(parents=True, exist_ok=True)

    # copy over context binary files
    has_cb_files = copy_context_bin_files(model_path, new_model_path.parent, saved_cb_files=saved_external_files)

    external_file_names = get_external_data_file_names(model_path)

    if not external_file_names:
        if force_external_data:
            # save the model with single external data file
            model_proto_to_file(onnx.load(model_path), new_model_path, save_as_external_data=True)
            return True

        # no external data, so we can just copy the model
        hardlink_copy_file(model_path, new_model_path)
        return has_cb_files or False

    if len(external_file_names) > 1:
        # save the model with single external data file
        model_proto_to_file(onnx.load(model_path), new_model_path, save_as_external_data=True)
        return True

    external_file_path = str(model_path.parent / external_file_names[0])
    if external_file_path in saved_external_files:
        # already saved, model will refer to the same file
        new_external_file_name = saved_external_files[external_file_path]
    else:
        new_external_file_name = f"{new_model_path.name}.data"
        # copy the external data file to the new location
        hardlink_copy_file(external_file_path, new_model_path.parent / new_external_file_name)
        # update the saved external files mapping
        saved_external_files[external_file_path] = new_external_file_name

    # change the external data location and save the model file
    model_proto = onnx.load(model_path, load_external_data=False)
    change_external_data_location(model_proto, new_external_file_name)
    model_proto_to_file(model_proto, new_model_path)
    return True


LORA_NAME_PATTERNS_TORCHSCRIPT = [
    f".*[./]{name}[./]{matmul}$"
    for name in ["default_0", "default_0_1", "default", "default_1", "lora_A", "lora_B"]
    for matmul in ["MatMul", "MatMul_Q4"]
]
LOHA_NAME_PATTERNS_TORCHSCRIPT = [
    f".*[./]{name}[./]default" for name in ["hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b"]
]

DORA_NAME_PATTERNS_TORCHSCRIPT = [
    f".*{pattern}$"
    for pattern in [
        "default/Div",
        "default/default/MatMul",
        "default/default_1/MatMul",
        "default/default/MatMul_Q4",
        "default/default_1/MatMul_Q4",
    ]
]

LORA_NAME_PATTERNS_DYNAMO = ["lora_A", "lora_B"]
LOHA_NAME_PATTERNS_DYNAMO = ["hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b"]
DORA_NAME_PATTERNS_DYNAMO = ["lora_A", "lora_B", "lora_magnitude_vector"]
PATTERN_MAP_DYNAMO = {
    AdapterType.LORA: LORA_NAME_PATTERNS_DYNAMO,
    AdapterType.DORA: DORA_NAME_PATTERNS_DYNAMO,
    AdapterType.LOHA: LOHA_NAME_PATTERNS_DYNAMO,
}


def model_has_adapters_from_dynamo(model_path: Union[str, Path], adapter_type: AdapterType = AdapterType.LORA) -> bool:
    """Check if the model has adapters.

    :param model_path: The path to the model.
    :return: True if the model has adapters, False otherwise.
    """
    ir_model = ir.load(model_path)
    return any(
        get_adapter_name(initializer, PATTERN_MAP_DYNAMO.get(adapter_type, []))
        for initializer in ir_model.graph.initializers.values()
    )


def get_adapter_name(initializer, patterns) -> str:
    """Get the adapter name from the initializer.

    If the model is exported by torch dynamo, the LORA adapter may be folded into a single tensor.
    In this case, the adapter name is stored in the metadata_props of the initializer.
    We look for the adapter name in both the initializer name and the metadata_props.
    If not found, return None.

    :param initializer: The initializer to check.
    :param patterns: The patterns to look for.
    :return: The adapter name if found, None otherwise.
    """
    adapter_name = None

    if any(x in initializer.name for x in patterns):
        adapter_name = initializer.name
    elif FOLDED_FROM_KEY in initializer.metadata_props:
        import ast

        folded_from_str = initializer.metadata_props[FOLDED_FROM_KEY]
        folded_from = set(ast.literal_eval(folded_from_str))
        adapter_name = next((s for s in folded_from if any(x in s for x in patterns)), None)
    return adapter_name


# TODO(jambayk): considering matching by subgraph pattern, more involved but more reliable
def model_has_adapters_from_torchscript(
    model_path: Union[str, Path], adapter_type: AdapterType = AdapterType.LORA
) -> bool:
    """Check if the model has adapters.

    :param model_path: The path to the model.
    :return: True if the model has adapters, False otherwise.
    """
    dag = OnnxDAG(onnx.load(model_path, load_external_data=False))
    if adapter_type == AdapterType.LOHA and is_loha_model(dag):
        return True
    else:
        for node_name in dag.get_node_names():
            op_type = dag.get_node_op_type(node_name)
            if (adapter_type == AdapterType.LORA and is_lora_node(op_type, node_name)) or (
                adapter_type == AdapterType.DORA and is_dora_node(op_type, node_name)
            ):
                return True
    return False


def is_dora_node(op_type: str, node_name: str) -> bool:
    return op_type in {"MatMul", "MatMulNBits", "Div"} and any(
        re.match(pattern, node_name) for pattern in DORA_NAME_PATTERNS_TORCHSCRIPT
    )


def is_lora_node(op_type: str, node_name: str) -> bool:
    return op_type in {"MatMul", "MatMulNBits"} and any(
        re.match(pattern, node_name) for pattern in LORA_NAME_PATTERNS_TORCHSCRIPT
    )


def is_loha_model(dag: OnnxDAG) -> bool:
    for graph in dag.graphs:
        for initializer in graph.initializer:
            if any(re.match(pattern, initializer.name) for pattern in LOHA_NAME_PATTERNS_TORCHSCRIPT):
                return True
    return False


def model_has_adapters(model_path: Union[str, Path], adapter_type: AdapterType = AdapterType.LORA) -> bool:
    return model_has_adapters_from_dynamo(model_path, adapter_type) or model_has_adapters_from_torchscript(
        model_path, adapter_type
    )


def _fix_output_shapes(model_proto: onnx.ModelProto):
    """Run shape inference on the model and update the output shapes to make them fixed."""
    from onnxruntime.tools.onnx_model_utils import is_fixed_size_tensor
    from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

    # use the onnxruntime shape inference tool since it can handle large models as well as contrib ops
    inferred_proto = SymbolicShapeInference.infer_shapes(model_proto, auto_merge=True, guess_output_rank=True)

    for idx, o in enumerate(model_proto.graph.output):
        if not is_fixed_size_tensor(o):
            new_o = inferred_proto.graph.output[idx]
            if is_fixed_size_tensor(new_o):
                o.type.tensor_type.shape.CopyFrom(new_o.type.tensor_type.shape)


def fix_dim_params(model_proto: onnx.ModelProto, dim_params: list[str], dim_values: list[int]):
    """Fix the dimension parameters in the model.

    :param dim_params: The dimension parameters to fix.
    :param dim_values: The values to set for the dimension parameters.
    """
    from onnxruntime.tools.onnx_model_utils import make_dim_param_fixed

    assert len(dim_params) == len(dim_values), "dim_params and dim_values must have the same number of elements."
    assert all(i >= 0 for i in dim_values), "dim_values must be all >= 0"

    for param, value in zip(dim_params, dim_values):
        make_dim_param_fixed(model_proto.graph, param, value)

    # update the output shapes to make them fixed
    _fix_output_shapes(model_proto)


def fix_input_shapes(model_proto: onnx.ModelProto, input_names: list[str], input_shapes: list[list[int]]):
    """Fix the input shapes in the model.

    :param input_names: The input names to fix.
    :param input_shapes: The shapes to set for the inputs.
    """
    from onnxruntime.tools.onnx_model_utils import make_input_shape_fixed

    assert len(input_names) == len(input_shapes), "input_names and input_shapes must have the same number of elements."
    assert all(all(i > 0 for i in shape) for shape in input_shapes), "input_shapes must be all > 0"

    for name, shape in zip(input_names, input_shapes):
        make_input_shape_fixed(model_proto.graph, name, shape)

    # update the output shapes to make them fixed
    _fix_output_shapes(model_proto)


def process_llm_pipeline(
    model: CompositeModelHandler,
    llm_pipeline: list,
    process_func: Callable,
    output_dir: Union[str, Path],
    decoder_config_extra: Optional[dict[str, Any]] = None,
    group_session_options: Optional[dict[str, Any]] = None,
) -> CompositeModelHandler:
    """Process an LLM pipeline with the given function.

    :param model_handler: The composite model with the pipeline.
    :param llm_pipeline: The pipeline to process.
    :param process_func: The function to apply to the context and iterator groups.
        Must accept a mapping from component name to component handler, pipeline, and output directory.
        Returns a
    :param output_dir: The directory to save the processed model.
    :param decoder_config_extra: Extra configuration for the decoder.
    :param group_session_options: Session options for the context and iterator groups.
    :return: The processed composite model handler.
    """
    output_dir = Path(output_dir)
    component_models = dict(model.get_model_components())

    new_component_models = {}
    new_llm_pipeline = {}

    # resave embeddings model
    embeddings_model_path = output_dir / "embeddings.onnx"
    resave_model(component_models[llm_pipeline["embeddings"]].model_path, embeddings_model_path)
    new_component_models["embeddings"] = ONNXModelHandler(
        model_path=output_dir, onnx_file_name=embeddings_model_path.name
    )
    new_llm_pipeline["embeddings"] = "embeddings"

    # process the context and iterator models
    new_groups = process_func(component_models, llm_pipeline, output_dir)
    for key, group in new_groups.items():
        new_component_models.update(group)
        new_llm_pipeline[key] = list(group.keys())

    # resave the lm_head model
    lm_head_model_path = output_dir / "lm_head.onnx"
    resave_model(component_models[llm_pipeline["lm_head"]].model_path, lm_head_model_path)
    new_component_models["lm_head"] = ONNXModelHandler(model_path=output_dir, onnx_file_name=lm_head_model_path.name)
    new_llm_pipeline["lm_head"] = "lm_head"

    # create new model attributes
    new_model_attributes = deepcopy(model.model_attributes) or {}
    new_model_attributes["llm_pipeline"] = new_llm_pipeline

    return update_llm_pipeline_genai_config(
        CompositeModelHandler(
            list(new_component_models.values()),
            list(new_component_models.keys()),
            model_path=output_dir,
            model_attributes=new_model_attributes,
        ),
        source_llm_pipeline=llm_pipeline,
        decoder_config_extra=decoder_config_extra,
        group_session_options=group_session_options,
    )


def update_llm_pipeline_genai_config(
    model: CompositeModelHandler,
    source_llm_pipeline: Optional[dict[str, Any]] = None,
    decoder_config_extra: Optional[dict[str, Any]] = None,
    group_session_options: Optional[dict[str, Any]] = None,
) -> CompositeModelHandler:
    """Update the LLM pipeline in the model's genai_config.json file.

    :param model: The composite model to update.
    :param source_llm_pipeline: The source LLM pipeline to use for the update.
    :param decoder_config_extra: Extra configuration for the decoder.
    :param group_session_options: Session options for the context and iterator groups.
    :return: The updated composite model.
    """
    if not model.model_path or not Path(model.model_path).is_dir():
        logger.warning("Model path is not set or is not a directory. Cannot update genai_config.json.")
        return model

    if not model.model_attributes or ({"llm_pipeline", "additional_files"} - model.model_attributes.keys()):
        # no llm_pipeline or additional_files, so just return the model
        return model

    additional_files = model.model_attributes["additional_files"]
    llm_pipeline = model.model_attributes["llm_pipeline"]

    # update genai_config if it exists
    genai_config_path = None
    for file_path in additional_files:
        if Path(file_path).name == "genai_config.json":
            genai_config_path = file_path
            break

    if not genai_config_path:
        # no genai_config, so just return the model
        return model

    with open(genai_config_path) as f:
        genai_config = json.load(f)

    # update model_type
    genai_config["model"]["type"] = "decoder-pipeline"

    # update decoder config
    decoder_config = genai_config["model"]["decoder"]
    decoder_config.pop("filename", None)
    # this option is used for a different type of sliding window in ort-genai 0.9.0+
    decoder_config.get("sliding_window", {}).pop("slide_inputs", None)
    for key, value in (decoder_config_extra or {}).items():
        exisiting_value = decoder_config.get(key)
        if isinstance(exisiting_value, dict):
            exisiting_value.update(value)
        elif isinstance(exisiting_value, list):
            exisiting_value.extend(value)
        else:
            decoder_config[key] = value

    # get group session options
    if source_llm_pipeline:
        group_session_options = group_session_options or decoder_config.get("pipeline", [{}])[0].get(
            source_llm_pipeline["context"][0], {}
        ).get("session_options")
    # update pipeline config
    component_models = dict(model.get_model_components())
    pipeline_config = {}
    for name in [
        llm_pipeline["embeddings"],
        *llm_pipeline["context"],
        *llm_pipeline["iterator"],
        llm_pipeline["lm_head"],
    ]:
        component = component_models[name]
        component_io_config = component.io_config
        pipeline_config[name] = {
            "filename": Path(component.model_path).name,
            "inputs": component_io_config["input_names"],
            "outputs": component_io_config["output_names"],
        }

    for group, dont_run_on in zip(["context", "iterator"], ["token_gen", "prompt"]):
        for name in llm_pipeline[group]:
            if group_session_options:
                pipeline_config[name]["session_options"] = group_session_options
            pipeline_config[name][f"run_on_{dont_run_on}"] = False

    pipeline_config[llm_pipeline["lm_head"]]["is_lm_head"] = True

    decoder_config["pipeline"] = [pipeline_config]

    # save the updated genai_config
    new_genai_config_path = Path(model.model_path) / "genai_config.json"
    with new_genai_config_path.open("w") as f:
        json.dump(genai_config, f, indent=4)
    # update the model attributes
    additional_files.remove(genai_config_path)
    additional_files.append(str(new_genai_config_path))

    return model


def update_llm_pipeline_genai_config_gpu(
    model: ONNXModelHandler,
    output_model_dir: Union[str, Path],
    input_model_path: Union[str, Path],
    decoder_config_extra: Optional[dict[str, Any]] = None,
) -> ONNXModelHandler:
    """Update the LLM pipeline in the model's genai_config.json file.

    :param model: The  model to update.
    :param decoder_config_extra: Extra configuration for the decoder.
    """
    output_model_dir = Path(output_model_dir)

    # update genai_config if it exists
    genai_config_path = None
    genai_config_path = Path(input_model_path).parent / "genai_config.json"

    if genai_config_path.exists():
        genai_config_path = str(genai_config_path.resolve())
    else:
        return model

    with open(genai_config_path) as f:
        genai_config = json.load(f)

    # update model_type
    genai_config["model"]["type"] = "decoder-pipeline"

    # Update the provider_options list
    provider_option = {"qnn": {"backend_type": "gpu"}}
    genai_config["model"]["decoder"]["session_options"]["provider_options"] = [provider_option]

    # update decoder config
    decoder_config = genai_config["model"]["decoder"]
    decoder_config.get("sliding_window", {}).pop("slide_inputs", None)
    for key, value in (decoder_config_extra or {}).items():
        exisiting_value = decoder_config.get(key)
        if isinstance(exisiting_value, dict):
            exisiting_value.update(value)
        elif isinstance(exisiting_value, list):
            exisiting_value.extend(value)
        else:
            decoder_config[key] = value

    pipeline_config = {}
    component_io_config = model.io_config
    pipeline_config["model_onnx"] = {
        "filename": Path(model.model_path).name,
        "inputs": component_io_config["input_names"],
        "outputs": component_io_config["output_names"],
    }

    decoder_config["pipeline"] = [pipeline_config]

    # save the updated genai_config
    new_genai_config_path = output_model_dir / "genai_config.json"
    with new_genai_config_path.open("w") as f:
        json.dump(genai_config, f, indent=4)

    return model


def update_llm_pipeline_genai_config_gpu_ctxbin(
    model_path: Union[str, Path],
) -> None:
    """Update the filename fields in the model's genai_config.json file from 'model' to 'model_ctx'.

    The genai_config.json file is updated in place in the model's directory.
    :param model_path: Path to the model file.
    """
    # Find genai_config in the model's directory
    model_dir = Path(model_path).parent
    genai_config_path = model_dir / "genai_config.json"

    if not genai_config_path.exists():
        return

    with open(genai_config_path) as f:
        genai_config = json.load(f)

    # Update decoder filename to 'model_ctx'
    if "decoder" in genai_config.get("model", {}):
        if "filename" in genai_config["model"]["decoder"]:
            genai_config["model"]["decoder"]["filename"] = "model/model_ctx.onnx"

        # Update filename in pipeline configuration
        decoder_config = genai_config["model"]["decoder"]
        if "pipeline" in decoder_config and isinstance(decoder_config["pipeline"], list):
            for pipeline_item in decoder_config["pipeline"]:
                if "model_onnx" in pipeline_item and "filename" in pipeline_item["model_onnx"]:
                    pipeline_item["model_onnx"]["filename"] = "model/model_ctx.onnx"

    # Save the updated genai_config back to the same location
    with genai_config_path.open("w") as f:
        json.dump(genai_config, f, indent=4)

# -------------------------------------------------------------------------
# Copyright (c) Intel Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import os
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Union

from olive.common.config_utils import validate_config
from olive.data.config import DataConfig
from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.model.handler import CompositeModelHandler, HfModelHandler, ONNXModelHandler, OpenVINOModelHandler
from olive.passes import Pass
from olive.passes.openvino.ov_utils import (
    IgnoreScopeTypeEnum,
    OVOptimumLibrary,
    _convert_to_enum,
    _validate_enum_value,
    create_genai_config,
    infer_library_name,
)
from olive.passes.pass_config import BasePassConfig, ParamCategory, PassConfigParam, get_user_script_data_config

logger = logging.getLogger(__name__)


def _convert_compress_config_enums(compress_config: dict) -> dict:
    """Convert compress_config enum values from strings to enum instances.

    Handles both strings and existing enum instances (pass through unchanged).
    This function should be called at the point of use to ensure enum values are
    properly converted, especially when validate_config() may have been bypassed
    (e.g., in unit tests with disable_search=True).

    Args:
        compress_config: The compress_config dictionary to convert.

    Returns:
        The compress_config with enum values converted.

    Raises:
        ImportError: If nncf is not installed.
        ValueError: If an enum value is invalid.

    """
    try:
        import nncf
    except ImportError:
        raise ImportError("Please install olive-ai[openvino] to use OpenVINO NNCF") from None

    if not compress_config:
        return compress_config

    if compress_config.get("mode") is not None:
        compress_config["mode"] = _convert_to_enum(
            compress_config["mode"],
            nncf.parameters.CompressWeightsMode,
            "mode",
        )
    if compress_config.get("sensitivity_metric") is not None:
        compress_config["sensitivity_metric"] = _convert_to_enum(
            compress_config["sensitivity_metric"],
            nncf.parameters.SensitivityMetric,
            "sensitivity_metric",
        )
    if compress_config.get("backup_mode") is not None:
        compress_config["backup_mode"] = _convert_to_enum(
            compress_config["backup_mode"],
            nncf.parameters.BackupMode,
            "backup_mode",
        )
    if compress_config.get("compression_format") is not None:
        compress_config["compression_format"] = _convert_to_enum(
            compress_config["compression_format"],
            nncf.parameters.CompressionFormat,
            "compression_format",
        )

    return compress_config


def _validate_advanced_compression_params(advanced_params: Optional[dict]) -> tuple[bool, str]:
    """Validate advanced_compression_parameters enum values.

    This is a validation-only function that does not modify the value.
    Use _get_advanced_compression_params for actual conversion.

    Args:
        advanced_params: The advanced_compression_parameters dictionary to validate.

    Returns:
        Tuple of (is_valid, error_message). error_message is empty if valid.

    """
    if not advanced_params:
        return True, ""

    # Import NNCF advanced parameter types for validation
    try:
        from nncf.quantization.advanced_parameters import GroupSizeFallbackMode
    except ImportError:
        return False, "Please install olive-ai[openvino] to use OpenVINO NNCF"

    # Validate group_size_fallback_mode if present
    if advanced_params.get("group_size_fallback_mode") is not None:
        is_valid, error_msg = _validate_enum_value(
            advanced_params["group_size_fallback_mode"],
            GroupSizeFallbackMode,
            "group_size_fallback_mode",
        )
        if not is_valid:
            return False, error_msg

    return True, ""


class OpenVINOWeightCompression(Pass):
    """OpenVINO weight compression pass.

    Please refer to https://docs.openvino.ai/2025/openvino-workflow/model-optimization-guide/weight-compression.html for more details.
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            **get_user_script_data_config(),
            "data_config": PassConfigParam(
                type_=Union[DataConfig, dict],
                required=False,
                description="Data config for compression.",
            ),
            "ignored_scope": PassConfigParam(
                type_=Union[list[str], list[list[str]]],
                required=False,
                default_value=None,
                description=(
                    "This parameter can be used to exclude some layers based on their names, types, and/or patterns "
                    "from the compression process to preserve the model accuracy. Please refer to "
                    "https://docs.openvino.ai/2025/openvino-workflow/model-optimization-guide/quantizing-models-post-training/basic-quantization-flow.html#tune-quantization-parameters."
                    "If multiple ignored_scope_types are provided, ignored_scope should be a list of lists, "
                    "where each inner list corresponds to a specific ignored_scope_type in the same order."
                ),
            ),
            "ignored_scope_type": PassConfigParam(
                type_=Union[IgnoreScopeTypeEnum, list[IgnoreScopeTypeEnum]],
                required=False,
                default_value=None,
                description="Defines the type(s) of the ignored scope. Supported values: 'names', 'types', 'patterns'.",
            ),
            "target_device": PassConfigParam(
                type_=Device,
                required=False,
                default_value=accelerator_spec.accelerator_type,
                description=(
                    "Target device for the model. "
                    "Supported values: 'any', 'cpu', 'gpu', 'cpu_spr', 'npu'. "
                    "Default value is the same as the accelerator type of this workflow run."
                ),
            ),
            "transform_fn": PassConfigParam(
                type_=Union[Callable, str],
                category=ParamCategory.OBJECT,
                required=False,
                description="Transform function for the input data.",
            ),
            "extra_args": PassConfigParam(
                type_=dict,
                required=False,
                default_value=None,
                description="Extra arguments to pass to the `nncf.compress_weights()` function.",
            ),
            "compress_config": PassConfigParam(
                type_=dict,
                required=False,
                default_value=None,
                description=(
                    "Weight Compression configuration for OpenVINO model weight compression. Please refer to "
                    "https://docs.openvino.ai/2025/openvino-workflow/model-optimization-guide/weight-compression.html."
                ),
            ),
            "reuse_cache": PassConfigParam(
                type_=bool,
                default_value=False,
                required=False,
                description=("Reuse cache of previous passes to reduce storage footprint."),
            ),
        }

    @classmethod
    def validate_config(
        cls,
        config: type[BasePassConfig],
        accelerator_spec: AcceleratorSpec,
    ) -> bool:
        try:
            import nncf
        except ImportError:
            raise ImportError("Please install nncf to use OpenVINO Weight Compression") from None

        if not super().validate_config(config, accelerator_spec):
            return False

        # Validate compress_config enum parameters
        if config.compress_config:
            enum_validations = [
                (config.compress_config.get("mode"), nncf.parameters.CompressWeightsMode, "mode"),
                (
                    config.compress_config.get("sensitivity_metric"),
                    nncf.parameters.SensitivityMetric,
                    "sensitivity_metric",
                ),
                (config.compress_config.get("backup_mode"), nncf.parameters.BackupMode, "backup_mode"),
                (
                    config.compress_config.get("compression_format"),
                    nncf.parameters.CompressionFormat,
                    "compression_format",
                ),
            ]
            for value, enum_class, param_name in enum_validations:
                is_valid, error_msg = _validate_enum_value(value, enum_class, param_name)
                if not is_valid:
                    logger.error(error_msg)
                    return False

        # Validate extra_args enum parameters
        if config.extra_args:
            extra_validations = [
                (config.extra_args.get("model_type"), nncf.ModelType, "model_type"),
                (config.extra_args.get("preset"), nncf.QuantizationPreset, "preset"),
                (config.extra_args.get("library"), OVOptimumLibrary, "library"),
            ]
            for value, enum_class, param_name in extra_validations:
                is_valid, error_msg = _validate_enum_value(value, enum_class, param_name)
                if not is_valid:
                    logger.error(error_msg)
                    return False

            # Validate advanced_compression_parameters
            is_valid, error_msg = _validate_advanced_compression_params(
                config.extra_args.get("advanced_compression_parameters")
            )
            if not is_valid:
                logger.error(error_msg)
                return False

        return True

    @staticmethod
    def _create_calibration_dataset(common_dataloader):
        """Create an nncf.Dataset instance from a common dataloader."""
        try:
            import nncf
        except ImportError:
            raise ImportError("Please install olive-ai[openvino] to use OpenVINO NNCF") from None

        def transform_fn(data_item):
            data, _ = data_item
            return data

        return nncf.Dataset(common_dataloader, transform_fn)

    def _get_nncf_dataset(self, config, tokenizer: Optional = None):
        try:
            import nncf
        except ImportError:
            raise ImportError("Please install olive-ai[openvino] to use OpenVINO NNCF") from None

        data_loader = None
        if config.data_config:
            data_config = validate_config(config.data_config, DataConfig)
            data_loader = data_config.to_data_container().create_dataloader()

        # if a data_config is not specified, return None
        if data_loader is None:
            return None

        def transform_fn(data_item):
            data, _ = data_item
            return data

        transform_func = (
            self._user_module_loader.load_object(config.transform_fn) if config.transform_fn else transform_fn
        )

        # use extra args to load tokenizer and pass via partial
        if config.extra_args and tokenizer is not None:
            transform_func = partial(transform_func, tokenizer=tokenizer)

        return nncf.Dataset(data_loader, transform_func)

    @staticmethod
    def _get_extra_params(config):
        """Get extra parameters for NNCF compression.

        Converts model_type and preset to enum values at point of use to handle cases
        where validate_config() may have been bypassed (e.g., in unit tests).

        Args:
            config: The pass configuration.

        Returns:
            Dictionary of extra parameters for NNCF compression.

        Raises:
            ImportError: If nncf is not installed.
            ValueError: If ignored_scope configuration is invalid, or if model_type/preset values are invalid.

        """
        try:
            import nncf
        except ImportError:
            raise ImportError("Please install olive-ai[openvino] to use OpenVINO NNCF") from None

        extra_params = {}
        # Convert model_type and preset to enums at point of use
        # (handles case where validate_config was bypassed, e.g., in unit tests)
        if config.extra_args and config.extra_args.get("model_type") is not None:
            extra_params["model_type"] = _convert_to_enum(
                config.extra_args.get("model_type"), nncf.ModelType, "model_type"
            )
        if config.extra_args and config.extra_args.get("preset") is not None:
            extra_params["preset"] = _convert_to_enum(
                config.extra_args.get("preset"), nncf.QuantizationPreset, "preset"
            )
        # target device is not needed for weight compression with NNCF
        if (config.ignored_scope and not config.ignored_scope_type) or (
            config.ignored_scope_type and not config.ignored_scope
        ):
            raise ValueError(
                "Both 'ignored_scope' and 'ignored_scope_type' must be provided together for ignored scope configuration."
            )
        if config.ignored_scope and config.ignored_scope_type:
            # Handle list of ignored_scope_types by zipping with ignored_scope.
            # Ensure ignored_scope is a list of lists if ignored_scope_type is a list
            # with number of elements in ignored_scope equalling number of elements in ignored_scope_type.
            if isinstance(config.ignored_scope_type, list):
                if isinstance(config.ignored_scope, list) and all(
                    isinstance(item, list) for item in config.ignored_scope
                ):
                    if len(set(config.ignored_scope_type)) != len(config.ignored_scope_type):
                        raise ValueError(
                            "All values in ignored_scope_type must be unique to avoid overwriting in the ignored_scope dictionary."
                        )
                    if len(config.ignored_scope) != len(config.ignored_scope_type):
                        raise ValueError(
                            "Length of ignored_scope must match length of ignored_scope_type when both are lists."
                        )
                    kwargs = dict(zip(config.ignored_scope_type, config.ignored_scope))
                else:
                    raise ValueError("When ignored_scope_type is a list, ignored_scope must be a list of lists.")
            else:
                kwargs = {config.ignored_scope_type: config.ignored_scope}
            extra_params["ignored_scope"] = nncf.IgnoredScope(**kwargs)

        return extra_params

    @staticmethod
    def _get_advanced_compression_params(config):
        """Get advanced compression parameters for NNCF.

        Converts group_size_fallback_mode to enum and nested dataclass parameters.

        Args:
            config: The pass configuration.

        Returns:
            Dictionary of advanced compression parameters for NNCF.

        Raises:
            ImportError: If nncf is not installed.
            ValueError: If group_size_fallback_mode value is invalid.

        """
        advanced_params = {}
        if config.extra_args and config.extra_args.get("advanced_compression_parameters") is not None:
            advanced_params = deepcopy(config.extra_args.get("advanced_compression_parameters"))

        if not advanced_params:
            return advanced_params

        # Import NNCF advanced parameter types
        try:
            from nncf.quantization.advanced_parameters import (
                AdvancedAWQParameters,
                AdvancedGPTQParameters,
                AdvancedLoraCorrectionParameters,
                AdvancedScaleEstimationParameters,
                GroupSizeFallbackMode,
            )
        except ImportError:
            raise ImportError("Please install olive-ai[openvino] to use OpenVINO NNCF") from None

        # Convert group_size_fallback_mode string to enum if present
        if advanced_params.get("group_size_fallback_mode") is not None:
            advanced_params["group_size_fallback_mode"] = _convert_to_enum(
                advanced_params["group_size_fallback_mode"],
                GroupSizeFallbackMode,
                "group_size_fallback_mode",
            )

        # Convert nested dataclass parameters if they are dicts
        if advanced_params.get("awq_params") is not None:
            awq_params = advanced_params.get("awq_params")
            if isinstance(awq_params, dict):
                advanced_params["awq_params"] = AdvancedAWQParameters(**awq_params)

        if advanced_params.get("scale_estimation_params") is not None:
            scale_params = advanced_params.get("scale_estimation_params")
            if isinstance(scale_params, dict):
                advanced_params["scale_estimation_params"] = AdvancedScaleEstimationParameters(**scale_params)

        if advanced_params.get("gptq_params") is not None:
            gptq_params = advanced_params.get("gptq_params")
            if isinstance(gptq_params, dict):
                advanced_params["gptq_params"] = AdvancedGPTQParameters(**gptq_params)

        if advanced_params.get("lora_correction_params") is not None:
            lora_params = advanced_params.get("lora_correction_params")
            if isinstance(lora_params, dict):
                advanced_params["lora_correction_params"] = AdvancedLoraCorrectionParameters(**lora_params)

        # Handle backend_params - extract external_dir for runtime processing
        # Note: backend_params is backend-specific (ONNX vs OpenVINO) and will be
        # converted at runtime using the appropriate BackendParameters class
        if advanced_params.get("backend_params") is not None:
            backend_params = advanced_params.get("backend_params")
            if isinstance(backend_params, dict):
                # Pop external_dir from backend_params - will be added at runtime
                external_dir = backend_params.pop("external_dir", None)
                if not backend_params:
                    # Remove empty backend_params after popping external_dir
                    advanced_params.pop("backend_params")
                # Store external_dir separately if it was present
                if external_dir is not None:
                    advanced_params["_external_dir"] = external_dir

        return advanced_params

    def _apply_compression(
        self,
        model_to_compress: Any,
        config: type[BasePassConfig],
        output_model_path: str,
        tokenizer: Optional[Any] = None,
    ) -> Any:
        """Apply NNCF weight compression to a model.

        Args:
            model_to_compress: The model object to compress (OpenVINO model or ONNX model).
            config: The pass configuration.
            output_model_path: Path where the output model will be saved.
            tokenizer: Optional tokenizer for dataset transform (used in HF path).

        Returns:
            The compressed model object from nncf.compress_weights().

        Raises:
            ImportError: If nncf is not installed.

        """
        try:
            import nncf
            from nncf.onnx.quantization.backend_parameters import BackendParameters
        except ImportError:
            raise ImportError("Please install olive-ai[openvino] to use OpenVINO NNCF") from None

        # get the weight compression dataset
        compression_dataset = self._get_nncf_dataset(config, tokenizer)

        # get the extra params
        extra_params = self._get_extra_params(config)

        # local copy of compress_config and ensure enum values are converted
        # (handles case where validate_config was bypassed, e.g., in unit tests)
        compress_config = deepcopy(config.compress_config) if config.compress_config else {}
        compress_config = _convert_compress_config_enums(compress_config)

        # append extra params to compress config
        compress_config.update(extra_params)

        # get nncf.AdvancedCompressionParameters if any
        advanced_params = None
        adv_par = self._get_advanced_compression_params(config)
        if adv_par is not None:
            # Handle external_dir for backend_params - add output path at runtime
            if adv_par.get("_external_dir") is not None:
                # Create or update backend_params with external data dir
                if adv_par.get("backend_params") is None:
                    adv_par["backend_params"] = {BackendParameters.EXTERNAL_DATA_DIR: output_model_path}
                else:
                    adv_par["backend_params"][BackendParameters.EXTERNAL_DATA_DIR] = output_model_path
                # Remove the temporary _external_dir key
                adv_par.pop("_external_dir")

            advanced_params = nncf.AdvancedCompressionParameters(**adv_par)

        # perform weight compression
        return nncf.compress_weights(
            model_to_compress, dataset=compression_dataset, advanced_parameters=advanced_params, **compress_config
        )

    def _run_for_config(
        self,
        model: Union[HfModelHandler, ONNXModelHandler, OpenVINOModelHandler],
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> Union[OpenVINOModelHandler, ONNXModelHandler, CompositeModelHandler]:
        if not isinstance(model, (HfModelHandler, ONNXModelHandler, OpenVINOModelHandler)):
            raise TypeError(
                "OpenVINOWeightCompression pass can only be applied to Hugging Face, ONNX, or OpenVINO models"
            )

        if config.reuse_cache:
            model_name_path = Path(model.model_path)
            weight_name_path = None
            if isinstance(model, OpenVINOModelHandler):
                model_name = model.model_config["model_name"]
                model_name_path = Path(model.model_path) / (f"{model_name}.xml")
                weight_name_path = Path(model.model_path) / (f"{model_name}.bin")
                output_model_path = model.model_path
            elif isinstance(model, ONNXModelHandler):
                output_model_path = str(
                    Path(model.model_path).with_name(Path(model.model_path).stem + "_compressed.onnx")
                )

        # initialize output_model to None
        output_model = None

        if isinstance(model, HfModelHandler):
            output_model = self._run_hf_pass(model, config, output_model_path)
        elif isinstance(model, ONNXModelHandler):
            output_model = self._run_onnx_pass(model, config, output_model_path)
        elif isinstance(model, OpenVINOModelHandler):
            output_model = self._run_openvino_pass(model, config, output_model_path)

        if config.reuse_cache:
            if os.path.exists(model_name_path):
                os.remove(model_name_path)
            if weight_name_path is not None and os.path.exists(weight_name_path):
                os.remove(weight_name_path)

        return output_model

    def _run_hf_pass(
        self,
        model: HfModelHandler,
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> Union[OpenVINOModelHandler, CompositeModelHandler]:
        try:
            from optimum.exporters.openvino import main_export as export_optimum_intel
        except ImportError:
            raise ImportError(
                "Please install Intel® optimum[openvino] to use NNCF for weight compression on HF models"
            ) from None

        # local copy of extra_args
        extra_args = deepcopy(config.extra_args) if config.extra_args else {}

        # set the library name for the HF Model
        if extra_args.get("library") is None:
            lib_name = infer_library_name(model.model_name_or_path)
        else:
            lib_name = extra_args["library"]

        # prepare extra args for export
        extra_args["stateful"] = not extra_args.get("disable_stateful", False)
        extra_args.pop("disable_stateful", None)
        extra_args["convert_tokenizer"] = not extra_args.get("disable_convert_tokenizer", False)
        extra_args.pop("disable_convert_tokenizer", None)
        extra_args["library_name"] = lib_name
        extra_args.pop("library", None)

        # export HF model to OpenVINO format
        export_optimum_intel(
            model.model_name_or_path,
            output_model_path,
            **extra_args,
        )

        # load the exported OpenVINO model
        from optimum.intel import OVModelForCausalLM

        output_model = OVModelForCausalLM.from_pretrained(output_model_path, compile=False)

        # redirect to ONNXModelHandler if extra_args requests ONNX processing
        # this is also only for CausalLM models
        if config.extra_args and config.extra_args.get("use_onnx") and isinstance(output_model, OVModelForCausalLM):
            try:
                from optimum.onnxruntime import ORTModelForCausalLM
            except ImportError:
                raise ImportError("Please install optimum[onnxruntime] to use ONNX models.") from None
            output_model = ORTModelForCausalLM.from_pretrained(model.model_name_or_path, export=True)

            # if pad_token_id is not set, set it to eos_token_id to avoid warnings during generation
            if output_model.config.pad_token_id == -1:
                output_model.config.pad_token_id = output_model.config.eos_token_id
                logger.warning(
                    "pad_token_id is not set. Setting pad_token_id to eos_token_id: %s to avoid warnings during generation.",
                    output_model.config.eos_token_id,
                )
            if output_model.generation_config.pad_token_id == -1:
                output_model.generation_config.pad_token_id = output_model.config.eos_token_id
                logger.warning(
                    "generation_config.pad_token_id is not set. Setting generation_config.pad_token_id to eos_token_id: %s to avoid warnings during generation.",
                    output_model.config.eos_token_id,
                )

            output_model.save_pretrained(output_model_path)
            omp = Path(output_model_path) / "model.onnx"
            omh = ONNXModelHandler(model_path=omp)
            return self._run_onnx_pass(omh, config, output_model_path)

        # initialize tokenizer to None
        tokenizer = None
        if config.extra_args and config.extra_args.get("tokenizer"):
            try:
                from transformers import AutoTokenizer
            except ImportError:
                raise ImportError(
                    "Install transformers to use NNCF for weight compression with tokenizers for Huggingface models"
                ) from None
            tokenizer = AutoTokenizer.from_pretrained(model.model_name_or_path)

        # perform weight compression using shared compression logic
        output_model.model = self._apply_compression(output_model.model, config, output_model_path, tokenizer)

        # save compressed model to temp directory to avoid file locking issues,
        # then copy back to output_model_path
        import gc
        import shutil
        import tempfile

        temp_dir = None
        try:
            temp_dir = tempfile.mkdtemp(prefix="olive_ov_compress_")
            output_model.save_pretrained(temp_dir)

            # release model to free file handles before copying
            del output_model
            gc.collect()

            # copy all files from temp_dir back to output_model_path
            for item in Path(temp_dir).iterdir():
                dest = Path(output_model_path) / item.name
                if item.is_file():
                    shutil.copy2(item, dest)
                elif item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
        finally:
            # clean up temp directory
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)

        # check the exported components
        exported_models = [name.stem for name in Path(output_model_path).iterdir() if name.suffix == ".xml"]
        logger.debug("Exported models are: %s.", exported_models)

        # OpenVINOModelHandler requires a directory with a single xml and bin file
        # OpenVINOModelHandler does not support multiple models in a single directory
        # If tokenizers are converted, those should be in a separate directory
        # OpenVINO would usually create both a tokenizer and a detokenizer in the same folder
        # return only the folder with just the OpenVINO model, not the tokenizer and detokenizer models.
        assert exported_models is not None
        assert len(exported_models) > 0, "No OpenVINO models were exported."

        # do not include tokenizer and detokenizer models for composite model creation
        remove_list = ["openvino_tokenizer", "openvino_detokenizer"]
        components = deepcopy(exported_models)
        if len(exported_models) > 1:
            for exported_model in exported_models:
                # move all extra OpenVINO XML and bin files to their respective subfolders
                if exported_model != "openvino_model":
                    extra_model_xml = Path(output_model_path) / f"{exported_model}.xml"
                    extra_model_bin = Path(output_model_path) / f"{exported_model}.bin"
                    dest_subdir = Path(output_model_path) / exported_model
                    dest_subdir.mkdir(parents=True, exist_ok=True)
                    if extra_model_xml.exists():
                        dest_xml = Path(dest_subdir) / f"{exported_model}.xml"
                        extra_model_xml.rename(dest_xml)
                        logger.debug("Moved %s to %s.", extra_model_xml, dest_xml)
                    if extra_model_bin.exists():
                        dest_bin = Path(dest_subdir) / f"{exported_model}.bin"
                        extra_model_bin.rename(dest_bin)
                        logger.debug("Moved %s to %s.", extra_model_bin, dest_bin)
                if exported_model in remove_list:
                    components.remove(exported_model)

        assert len(components) > 0, "No OpenVINO models were exported."

        # if only one model was exported return it directly
        if len(components) == 1:
            # will always return an OpenVINO model handler with folder as the model path
            return OpenVINOModelHandler(model_path=output_model_path)

        # if there are multiple components, return a composite model
        model_components = []
        model_component_names = []
        for component_name in components:
            if component_name in remove_list:
                # skip tokenizer and detokenizer models from the composite model
                continue
            if component_name != "openvino_model":
                # Each component is in a separate subfolder
                model_components.append(OpenVINOModelHandler(model_path=Path(output_model_path) / component_name))
            else:
                # The main model is in the output_model_path
                model_components.append(OpenVINOModelHandler(model_path=output_model_path))
            model_component_names.append(component_name)
        return CompositeModelHandler(model_components, model_component_names, model_path=output_model_path)

    def _run_onnx_pass(
        self,
        model: ONNXModelHandler,
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> ONNXModelHandler:
        try:
            import onnx
        except ImportError:
            raise ImportError(
                "Please install Intel® NNCF and ONNX to use nncf.compress_weights() on ONNX models"
            ) from None

        # load model
        loaded_model = onnx.load(model.model_path, load_external_data=False)

        # convert model to target opset version if necessary
        target_opset = 21 if config.extra_args is None else config.extra_args.get("target_opset", 21)
        if loaded_model.opset_import[0].version != target_opset:
            loaded_model = onnx.version_converter.convert_version(loaded_model, target_opset)

        # perform weight compression using shared compression logic
        output_model = self._apply_compression(loaded_model, config, output_model_path)

        # save to output_model_path
        model_name = Path(model.model_path).name.replace(".onnx", "_compressed.onnx")
        model_dir = Path(output_model_path)

        if Path(output_model_path).is_dir():
            output_model_path = Path(output_model_path) / model_name
        onnx.save(output_model, output_model_path, save_as_external_data=True)

        # generate the genai_config.json file for GenAI ONNX models
        create_genai_config(model_name, model_dir, config)

        return ONNXModelHandler(model_path=output_model_path)

    def _run_openvino_pass(
        self,
        model: OpenVINOModelHandler,
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> OpenVINOModelHandler:
        """Run weight compression on an OpenVINO model.

        Args:
            model: The OpenVINO model handler.
            config: The pass configuration.
            output_model_path: Path where the output model will be saved.

        Returns:
            OpenVINOModelHandler for the compressed model.

        Raises:
            ImportError: If openvino is not installed.

        """
        try:
            import openvino as ov
        except ImportError:
            raise ImportError("Please install openvino to use OpenVINO weight compression") from None

        # load the OpenVINO model
        core = ov.Core()
        model_config = model.model_config
        loaded_model = core.read_model(model_config["model"])

        # perform weight compression using shared compression logic
        compressed_model = self._apply_compression(loaded_model, config, output_model_path)

        # save the compressed model
        output_dir = Path(output_model_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_name = model_config["model_name"]
        output_xml_path = output_dir / f"{model_name}.xml"
        ov.save_model(compressed_model, output_xml_path)

        return OpenVINOModelHandler(model_path=output_model_path)

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import os
from pathlib import Path
from typing import Callable, Union

from olive.common.config_utils import validate_config
from olive.common.utils import StrEnumBase, hardlink_copy_dir, hardlink_copy_file
from olive.data.config import DataConfig
from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.model import OliveModelHandler
from olive.model.handler import ONNXModelHandler, OpenVINOModelHandler
from olive.passes import Pass
from olive.passes.onnx.common import model_proto_to_file
from olive.passes.openvino.ov_utils import IgnoreScopeTypeEnum
from olive.passes.pass_config import BasePassConfig, ParamCategory, PassConfigParam, get_user_script_data_config

logger = logging.getLogger(__name__)


def _default_validate_func(model, validation_loader) -> float:
    try:
        import numpy as np
        from onnx import ModelProto
        from openvino import CompiledModel
        from sklearn.metrics import accuracy_score
    except ImportError:
        raise ImportError("Please install olive-ai[openvino] and onnx to use NNCF quantization") from None

    predictions = []
    references = []

    if isinstance(model, CompiledModel):
        model_outputs = model.outputs
    elif isinstance(model, ModelProto):
        # instantiate ort session
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "Please install onnxruntime to use NNCF quantization with accuracy for ONNX models"
            ) from None
        ort_session = ort.InferenceSession(model.SerializeToString())
        model_inputs = ort_session.get_inputs()
        assert len(model_inputs) == 1, (
            "Model should have exactly one input for default validation function. Define your own validation function if it has more than one input."
        )
        model_outputs = ort_session.get_outputs()
        assert len(model_outputs) == 1, (
            "The model has multiple outputs, which is not supported by the default validation function. "
            "Please define a custom validation function to handle models with multiple outputs. "
        )
    else:
        raise TypeError("Model should be either CompiledModel or ModelProto.")

    output = model_outputs[0]

    try:
        import torch
    except ImportError:
        raise ImportError("Please install torch to use NNCF quantization with accuracy") from None

    for data_item, target in validation_loader:
        if isinstance(model, CompiledModel):
            pred = model(data_item)[output]
        else:
            input_name = model_inputs[0].name
            input_dict = {input_name: data_item.numpy() if isinstance(data_item, torch.Tensor) else data_item}
            pred = ort_session.run([output.name], input_dict)
        cur_pred = None
        if isinstance(pred, list) and len(pred) == 1:
            cur_pred = np.argmax(pred[0], axis=1)
        elif isinstance(pred, np.ndarray):
            cur_pred = np.argmax(pred, axis=1)
        predictions.append(cur_pred)
        references.append(target.numpy() if isinstance(target, torch.Tensor) else target)

    predictions = np.concatenate(predictions, axis=0)
    references = np.concatenate(references, axis=0)
    return accuracy_score(predictions, references)


class ModelTypeEnum(StrEnumBase):
    TRANSFORMER = "TRANSFORMER"


class PresetEnum(StrEnumBase):
    PERFORMANCE = "PERFORMANCE"
    MIXED = "MIXED"


class DropTypeEnum(StrEnumBase):
    ABSOLUTE = "ABSOLUTE"
    RELATIVE = "RELATIVE"


class OpenVINOQuantizationBase(Pass):
    """Post-training quantization for OpenVINO model.

    Please refer to https://docs.openvino.ai/2023.3/ptq_introduction.html for more details.
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            **get_user_script_data_config(),
            "data_config": PassConfigParam(
                type_=Union[DataConfig, dict],
                required=True,
                description="Data config for calibration.",
            ),
            "model_type": PassConfigParam(
                type_=ModelTypeEnum,
                required=False,
                default_value=None,
                description=(
                    "Used to specify quantization scheme required for specific type of the model. "
                    "'TRANSFORMER' is the only supported special quantization scheme to preserve accuracy "
                    "after quantization of Transformer models (BERT, DistilBERT, etc.). None is default."
                ),
            ),
            "preset": PassConfigParam(
                type_=PresetEnum,
                required=False,
                default_value=PresetEnum.PERFORMANCE,
                description="Defines quantization scheme for the model. Supported values: 'PERFORMANCE', 'MIXED'.",
            ),
            "ignored_scope": PassConfigParam(
                type_=Union[list[str], list[list[str]]],
                required=False,
                default_value=None,
                description=(
                    "This parameter can be used to exclude some layers based on their names, types, and/or patterns "
                    "from the quantization process to preserve the model accuracy. Please refer to "
                    "https://docs.openvino.ai/2023.3/basic_quantization_flow.html#tune-quantization-parameters."
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
            "validation_func": PassConfigParam(
                type_=Union[Callable, str],
                required=False,
                category=ParamCategory.OBJECT,
                description=(
                    "Used to compute accuracy metric. "
                    "Validation function receives openvino.runtime.CompiledModel object "
                    " or onnxruntime.OnnxModel object "
                    "and validation dataloader and returns accuracy metric value."
                ),
            ),
            "max_drop": PassConfigParam(
                type_=float,
                default_value=0.01,
                description=(
                    "Defines the accuracy drop threshold. The quantization process stops "
                    "when the degradation of accuracy metric on the validation dataset is less than the max_drop. "
                    "NNCF will stop the quantization and report an error if the max_drop value can't be reached. "
                    "The default value is 0.01."
                ),
            ),
            "drop_type": PassConfigParam(
                type_=DropTypeEnum,
                required=False,
                default_value=DropTypeEnum.ABSOLUTE,
                description=(
                    "Defines the type of the max_drop. Supported values: 'ABSOLUTE', 'RELATIVE'. "
                    "The default value is 'ABSOLUTE'."
                ),
            ),
            "extra_configs": PassConfigParam(
                type_=list[dict],
                required=False,
                description=(
                    "Extra configurations for OpenVINO model quantization. Please refer to "
                    "https://docs.openvino.ai/2023.3/basic_quantization_flow.html#tune-quantization-parameters."
                ),
            ),
            "reuse_cache": PassConfigParam(
                type_=bool,
                default_value=False,
                required=False,
                description=("Reuse cache of previous passes to reduce storage footprint."),
            ),
        }

    @staticmethod
    def _create_calibration_dataset(common_dataloader):
        """Create an nncf.Dataset instance from a common dataloader."""
        try:
            import nncf
        except ImportError:
            raise ImportError("Please install olive-ai[openvino] to use OpenVINO pass") from None

        def transform_fn(data_item):
            data, _ = data_item
            return data

        return nncf.Dataset(common_dataloader, transform_fn)

    def _get_nncf_dataset(self, config):
        try:
            import nncf
        except ImportError:
            raise ImportError("Please install olive-ai[openvino] to use OpenVINO pass") from None

        data_loader = None
        if config.data_config:
            data_config = validate_config(config.data_config, DataConfig)
            data_loader = data_config.to_data_container().create_dataloader()

        def transform_fn(data_item):
            data, _ = data_item
            return data

        transform_func = (
            self._user_module_loader.load_object(config.transform_fn) if config.transform_fn else transform_fn
        )

        return nncf.Dataset(data_loader, transform_func)

    @staticmethod
    def _get_extra_params(config):
        import nncf

        device_map = {
            "cpu": nncf.TargetDevice.CPU,
            "gpu": nncf.TargetDevice.GPU,
            "cpu_spr": nncf.TargetDevice.CPU_SPR,
            "npu": nncf.TargetDevice.NPU,
            "any": nncf.TargetDevice.ANY,
        }

        extra_params = {}
        extra_params["model_type"] = nncf.ModelType.TRANSFORMER if config.model_type == "TRANSFORMER" else None
        extra_params["preset"] = (
            nncf.QuantizationPreset.PERFORMANCE if config.preset == "PERFORMANCE" else nncf.QuantizationPreset.MIXED
        )
        extra_params["target_device"] = device_map.get(config.target_device, nncf.TargetDevice.ANY)

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


class OpenVINOQuantization(OpenVINOQuantizationBase):
    def _run_for_config(
        self, model: Union[OpenVINOModelHandler, ONNXModelHandler], config: type[BasePassConfig], output_model_path: str
    ) -> Union[OpenVINOModelHandler, ONNXModelHandler]:
        if not isinstance(model, (OpenVINOModelHandler, ONNXModelHandler)):
            raise TypeError("OpenVINOQuantization pass can only be applied to OpenVINO or ONNX models.")
        if config.reuse_cache:
            if isinstance(model, OpenVINOModelHandler):
                model_name = model.model_config["model_name"]
                model_name_path = Path(model.model_path) / (f"{model_name}.xml")
                output_model_path = model.model_path
                weight_name_path = Path(model.model_path) / (f"{model_name}.bin")
            else:
                model_name_path = Path(model.model_path)
                output_model_path = "".join(model.model_path.split(".onnx")[:-1]) + "_quant.onnx"
                weight_name_path = None

        self._run_pass(model, config, output_model_path)

        if config.reuse_cache:
            if os.path.exists(model_name_path):
                os.remove(model_name_path)
            if weight_name_path is not None and os.path.exists(weight_name_path):
                os.remove(weight_name_path)

        # If the model is OpenVINO, return OpenVINOModelHandler, otherwise return ONNXModelHandler
        if isinstance(model, OpenVINOModelHandler):
            output_model = OpenVINOModelHandler(model_path=output_model_path)
        else:
            if Path(output_model_path).is_file():
                assert Path(output_model_path).suffix == ".onnx", (
                    f"Output model path {output_model_path} should have ONNX suffix."
                )
            if Path(output_model_path).is_dir():
                output_model_path = Path(output_model_path) / Path(model.model_path).name.replace(
                    ".onnx", "_quant.onnx"
                )
            output_model = ONNXModelHandler(model_path=output_model_path)

        return output_model

    def _run_pass(
        self, model: Union[OpenVINOModelHandler, ONNXModelHandler], config: type[BasePassConfig], output_model_path: str
    ):
        try:
            import nncf
            import openvino as ov
        except ImportError:
            raise ImportError("Please install olive-ai[openvino] to use OpenVINO model") from None

        calibration_dataset = self._get_nncf_dataset(config)
        loaded_model = model.load_model()
        extra_params = self._get_extra_params(config)

        # nncf.AdvancedQuantizationParameters
        advanced_params = None
        if config.extra_configs:
            for extra_config in config.extra_configs:
                if extra_config.get("advanced_quantization_parameters"):
                    advanced_params = nncf.AdvancedQuantizationParameters(
                        **extra_config["advanced_quantization_parameters"]
                    )

        quantized_model = nncf.quantize(
            loaded_model, calibration_dataset, advanced_parameters=advanced_params, **extra_params
        )

        if not config.reuse_cache:
            if isinstance(model, OpenVINOModelHandler):
                # copy JSON and text files for genai models
                all_genai_files = [
                    name for name in Path(model.model_path).iterdir() if name.suffix in [".json", ".txt"]
                ]
                for genai_file in all_genai_files:
                    src_pth = Path(model.model_path) / genai_file
                    dest_path = Path(output_model_path)
                    hardlink_copy_file(src_pth, dest_path, follow_symlinks=True)

                # copy tokenizer folder if it exists
                src_tokenizer = Path(model.model_path) / "openvino_tokenizer"
                if src_tokenizer.exists() and src_tokenizer.is_dir():
                    dest_tokenizer = Path(output_model_path) / "openvino_tokenizer"
                    hardlink_copy_dir(src_tokenizer, dest_tokenizer, symlinks=True)

                # copy detokenizer folder if it exists
                src_detokenizer = Path(model.model_path) / "openvino_detokenizer"
                if src_detokenizer.exists() and src_detokenizer.is_dir():
                    dest_detokenizer = Path(output_model_path) / "openvino_detokenizer"
                    hardlink_copy_dir(src_detokenizer, dest_detokenizer, symlinks=True)
            else:
                # For ONNX models, all files are in the same folder as the ONNX model file.
                # extract directory from file path
                if Path(model.model_path).is_file():
                    model_dir = Path(model.model_path).parent
                else:
                    model_dir = Path(model.model_path)
                if Path(output_model_path).is_file():
                    dest_dir = Path(output_model_path).parent
                else:
                    dest_dir = Path(output_model_path)
                hardlink_copy_dir(model_dir, dest_dir, symlinks=True)

        if isinstance(model, OpenVINOModelHandler):
            model_name = model.model_config["model_name"]
            output_model_name = f"{model_name}_quant.xml"
            output_model_path = Path(output_model_path) / output_model_name
            ov.save_model(quantized_model, output_model=output_model_path)
            del quantized_model
            del loaded_model
        else:
            if Path(output_model_path).is_dir():
                output_model_path = Path(output_model_path) / Path(model.model_path).name.replace(
                    ".onnx", "_quant.onnx"
                )
            model_proto_to_file(quantized_model, output_model_path)


class OpenVINOQuantizationWithAccuracy(OpenVINOQuantizationBase):
    def _run_for_config(
        self, model: OliveModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> OliveModelHandler:
        if not isinstance(model, (OpenVINOModelHandler, ONNXModelHandler)):
            raise TypeError("OpenVINOQuantizationWithAccuracy pass can only be applied to OpenVINO or ONNX models.")
        if config.reuse_cache:
            if isinstance(model, OpenVINOModelHandler):
                output_model_path = model.model_path
                model_name = model.model_config["model_name"]
                model_name_path = Path(model.model_path) / (f"{model_name}.xml")
                weight_name_path = Path(model.model_path) / (f"{model_name}.bin")
            else:
                model_name_path = Path(model.model_path)
                output_model_path = "".join(model.model_path.split(".onnx")[:-1]) + "_quant.onnx"
                weight_name_path = None

        self._run_pass(model, config, output_model_path)

        if config.reuse_cache:
            if os.path.exists(model_name_path):
                os.remove(model_name_path)
            if weight_name_path is not None and os.path.exists(weight_name_path):
                os.remove(weight_name_path)

        # If the model is OpenVINO, return OpenVINOModelHandler, otherwise return ONNXModelHandler
        if isinstance(model, OpenVINOModelHandler):
            output_model = OpenVINOModelHandler(model_path=output_model_path)
        else:
            if Path(output_model_path).is_file():
                assert Path(output_model_path).suffix == ".onnx", (
                    f"Output model path {output_model_path} should have ONNX suffix."
                )
            if Path(output_model_path).is_dir():
                output_model_path = Path(output_model_path) / Path(model.model_path).name.replace(
                    ".onnx", "_quant.onnx"
                )
            output_model = ONNXModelHandler(model_path=output_model_path)
        return output_model

    def _run_pass(self, model: OliveModelHandler, config: type[BasePassConfig], output_model_path: str):
        try:
            import nncf
            import openvino as ov
        except ImportError:
            raise ImportError("Please install olive-ai[openvino] to use OpenVINO model") from None

        calibration_dataset = self._get_nncf_dataset(config)
        validation_dataset = self._get_nncf_dataset(config)

        loaded_model = model.load_model()
        extra_params = self._get_extra_params(config)

        validate_func = (
            self._user_module_loader.load_object(config.validation_func)
            if config.validation_func
            else _default_validate_func
        )

        drop_type = nncf.DropType.ABSOLUTE if config.drop_type == "ABSOLUTE" else nncf.DropType.RELATIVE

        quantized_model = nncf.quantize_with_accuracy_control(
            loaded_model,
            calibration_dataset=calibration_dataset,
            validation_dataset=validation_dataset,
            validation_fn=validate_func,
            max_drop=config.max_drop,
            drop_type=drop_type,
            **extra_params,
        )

        if not config.reuse_cache:
            if isinstance(model, OpenVINOModelHandler):
                # copy JSON and text files for genai models
                all_genai_files = [
                    name for name in Path(model.model_path).iterdir() if name.suffix in [".json", ".txt"]
                ]
                for genai_file in all_genai_files:
                    src_pth = Path(model.model_path) / genai_file
                    dest_path = Path(output_model_path)
                    hardlink_copy_file(src_pth, dest_path, follow_symlinks=True)

                # copy tokenizer folder if it exists
                src_tokenizer = Path(model.model_path) / "openvino_tokenizer"
                if src_tokenizer.exists() and src_tokenizer.is_dir():
                    dest_tokenizer = Path(output_model_path) / "openvino_tokenizer"
                    hardlink_copy_dir(src_tokenizer, dest_tokenizer, symlinks=True)

                # copy detokenizer folder if it exists
                src_detokenizer = Path(model.model_path) / "openvino_detokenizer"
                if src_detokenizer.exists() and src_detokenizer.is_dir():
                    dest_detokenizer = Path(output_model_path) / "openvino_detokenizer"
                    hardlink_copy_dir(src_detokenizer, dest_detokenizer, symlinks=True)
            else:
                # For ONNX models, all files are in the same folder as the ONNX model file.
                # extract directory from file path
                if Path(model.model_path).is_file():
                    model_dir = Path(model.model_path).parent
                else:
                    model_dir = Path(model.model_path)
                if Path(output_model_path).is_file():
                    dest_dir = Path(output_model_path).parent
                else:
                    dest_dir = Path(output_model_path)
                hardlink_copy_dir(model_dir, dest_dir, symlinks=True)

        if isinstance(model, OpenVINOModelHandler):
            model_name = model.model_config["model_name"]
            output_model_name = f"{model_name}_quant.xml"
            output_model_path = Path(output_model_path) / output_model_name
            ov.save_model(quantized_model, output_model=output_model_path)
            del quantized_model
            del loaded_model
        else:
            if Path(output_model_path).is_dir():
                output_model_path = Path(output_model_path) / Path(model.model_path).name.replace(
                    ".onnx", "_quant.onnx"
                )
            model_proto_to_file(quantized_model, output_model_path)

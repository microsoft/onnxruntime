# -------------------------------------------------------------------------
# Copyright (c) Intel Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Union

from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE

from olive.common.utils import StrEnumBase
from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.model import CompositeModelHandler, HfModelHandler, OpenVINOModelHandler
from olive.passes import Pass
from olive.passes.openvino.ov_utils import OVOptimumLibrary, infer_library_name
from olive.passes.pass_config import BasePassConfig, PassConfigParam, get_user_script_data_config

logger = logging.getLogger(__name__)


def infer_task(
    task,
    model_name_or_path,
    subfolder: str = "",
    revision: Optional[str] = None,
    cache_dir: str = HUGGINGFACE_HUB_CACHE,
    token: Optional[Union[bool, str]] = None,
    library_name: Optional[str] = None,
    trust_remote_code: bool = False,
):
    try:
        from optimum.exporters.tasks import TasksManager
    except Exception as e:
        raise ImportError("Unable to import optimum packages:", e) from None

    try:
        from requests.exceptions import ConnectionError as RequestsConnectionError
    except Exception as e:
        raise ImportError("Unable to import ConnectionError packages:", e) from None

    original_task = task
    task = TasksManager.map_from_synonym(task)
    if task == "auto":
        if library_name == "open_clip":
            task = "zero-shot-image-classification"
        else:
            try:
                task = TasksManager._infer_task_from_model_name_or_path(  # pylint: disable=W0212
                    model_name_or_path=model_name_or_path,
                    subfolder=subfolder,
                    revision=revision,
                    cache_dir=cache_dir,
                    token=token,
                    library_name=library_name,
                )
            except KeyError as e:
                try:
                    from transformers import AutoConfig
                except ImportError as ie:
                    raise ImportError(f"Unable to import AutoConfig from transformers: {ie}") from None
                try:
                    config = AutoConfig.from_pretrained(model_name_or_path)
                    with_past_arch_list = ["MistralForCausalLM", "Zamba2ForCausalLM"]
                    architectures = getattr(config, "architectures", None) or []
                    if any(arch in architectures for arch in with_past_arch_list):
                        task = "text-generation-with-past"
                except Exception:
                    raise KeyError(
                        f"The task could not be automatically inferred. Please provide the argument --task with the relevant task from {', '.join(TasksManager.get_all_tasks())}. Detailed error: {e}"
                    ) from None
            except RequestsConnectionError as e:
                raise RequestsConnectionError(
                    f"The task could not be automatically inferred as this is available only for models hosted on the Hugging Face Hub. Please provide the argument --task with the relevant task from {', '.join(TasksManager.get_all_tasks())}. Detailed error: {e}"
                ) from None

    if library_name == "transformers":
        try:
            from transformers import AutoConfig
        except ImportError as e:
            raise ImportError(f"Unable to import AutoConfig from transformers: {e}") from None
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            trust_remote_code=trust_remote_code,
        )
        if hasattr(config, "export_model_type"):
            model_type = config.export_model_type
        else:
            model_type = config.model_type
        custom_architecture = model_type not in TasksManager._SUPPORTED_MODEL_TYPE  # pylint: disable=W0212
        if not custom_architecture and task + "-with-past" in TasksManager.get_supported_tasks_for_model_type(
            model_type, exporter="openvino", library_name=library_name
        ):
            # Make -with-past the default if --task was not explicitly specified
            if original_task == "auto":
                task = task + "-with-past"
            else:
                logger.info(
                    "The task `%s` was manually specified, and past key values will not be reused in the decoding."
                    " if needed, please pass `--task %s-with-past` to export using the past key values.",
                    task,
                    task,
                )
    return task


def _main_quantize(
    model_name_or_path: str,
    task: str,
    library_name: str,
    quantization_config: Union[dict, "OVQuantizationConfigBase"],  # noqa: F821
    output: Path,
    cache_dir: str,
    trust_remote_code: bool = False,
    subfolder: str = "",
    revision: str = "main",
    token: Optional[Union[bool, str]] = None,
    model_kwargs: Optional[dict[str, Any]] = None,
):
    try:
        from optimum.intel.openvino.utils import _HEAD_TO_AUTOMODELS
        from optimum.intel.utils.import_utils import is_diffusers_available
    except ImportError as e:
        raise ImportError("Please install Intel® optimum[openvino] to use OpenVINO Optimum Conversion") from e

    # Step 0. Infer task and library name if needed
    original_task = task
    task = infer_task(
        task,
        model_name_or_path,
        subfolder=subfolder,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        library_name=library_name,
        trust_remote_code=trust_remote_code,
    )
    if library_name is None:
        library_name = infer_library_name(
            model_name_or_path,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
        )

    # Step 1. Obtain the correct OpenVINO model class
    if library_name == "diffusers":
        if not is_diffusers_available():
            raise ValueError("Export of diffusers models requires the diffusers library to be installed.")

        try:
            from diffusers import DiffusionPipeline
        except ImportError as e:
            raise ImportError("Unable to import diffusers packages:", e) from None

        diffusers_config = DiffusionPipeline.load_config(model_name_or_path)
        class_name = diffusers_config.get("_class_name", None)
        ov_class_name = f"OV{class_name}"
        try:
            model_cls = getattr(__import__("optimum.intel", fromlist=[ov_class_name]), ov_class_name)
        except (AttributeError, ImportError) as e:
            raise RuntimeError(f"Wasn't able to locate OpenVINO class for {class_name} diffusion model.") from e
    else:
        try:
            model_cls_name = _HEAD_TO_AUTOMODELS[task.replace("-with-past", "")]
            if library_name == "sentence_transformers":
                model_cls_name = "OVSentenceTransformer"
            model_cls = getattr(__import__("optimum.intel", fromlist=[model_cls_name]), model_cls_name)
        except (AttributeError, ImportError, KeyError) as e:
            raise RuntimeError(f"Wasn't able to locate OpenVINO class for task {original_task} ({task}).") from e

    # Step 2. Load the exported model
    # Filter out keys that are explicitly passed to from_pretrained to avoid
    # "got multiple values for keyword argument" TypeError
    _explicit_keys = {"trust_remote_code", "cache_dir", "use_cache", "compile"}
    filtered_kwargs = {k: v for k, v in (model_kwargs or {}).items() if k not in _explicit_keys}
    model = model_cls.from_pretrained(
        output,
        compile=False,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
        use_cache=task.endswith("with-past"),
        **filtered_kwargs,
    )

    # Step 3. Apply quantization and save the quantized model
    model._apply_quantization(  # pylint: disable=W0212
        quantization_config,
        compile_only=False,
        compile_model=False,
        model_name_or_path=model_name_or_path,
        trust_remote_code=trust_remote_code,
        save_directory=output,
        immediate_save=True,
    )


class OVQuantMode(StrEnumBase):
    INT8 = "int8"
    F8E4M3 = "f8e4m3"
    F8E5M2 = "f8e5m2"
    NF4_F8E4M3 = "nf4_f8e4m3"
    NF4_F8E5M2 = "nf4_f8e5m2"
    INT4_F8E4M3 = "int4_f8e4m3"
    INT4_F8E5M2 = "int4_f8e5m2"


class OVOptimumFramework(StrEnumBase):
    PT = "pt"
    TF = "tf"


class OVWeightFormat(StrEnumBase):
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"
    MXFP4 = "mxfp4"
    NF4 = "nf4"


class OpenVINOOptimumConversion(Pass):
    """Convert a Hugging Face PyTorch model to OpenVINO model using the Optimum export function."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            **get_user_script_data_config(),
            "components": PassConfigParam(
                type_=list[str],
                default_value=None,
                description=(
                    "List of component models to export. E.g. ['decoder_model', 'decoder_with_past_model']. None means"
                    " export all components."
                ),
            ),
            "device": PassConfigParam(
                type_=Device,
                default_value=accelerator_spec.accelerator_type.CPU,
                description=(
                    "The device to use to do the export. Defaults to 'cpu'."
                    "This is the parameter that is directly passed to Optimum Intel export function in certain cases."
                ),
            ),
            "extra_args": PassConfigParam(
                type_=dict,
                default_value=None,
                description="Extra arguments to pass to the `optimum.exporters.openvino.main_export` function.",
            ),
            "ov_quant_config": PassConfigParam(
                type_=dict,
                default_value=None,
                required=False,
                description=(
                    "Parameters for optimum OpenVINO quantization. Please refer to "
                    "https://huggingface.co/docs/optimum/main/intel/openvino/optimization#4-bit"
                ),
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
            raise ImportError("Please install nncf to use OpenVINO Optimum Conversion") from None
        if not super().validate_config(config, accelerator_spec):
            return False

        # validate allowed libraries in extra_args if provided
        if (
            config.extra_args
            and config.extra_args.get("library") is not None
            and config.extra_args.get("library") not in [lib.value for lib in OVOptimumLibrary]
        ):
            logger.error(
                "Library %s is not supported. Supported libraries are %s.",
                config.extra_args.get("library"),
                ", ".join([lib.value for lib in OVOptimumLibrary]),
            )
            return False

        # validate allowed frameworks if provided
        if (
            config.extra_args
            and config.extra_args.get("framework") is not None
            and config.extra_args.get("framework") not in [framework.value for framework in OVOptimumFramework]
        ):
            logger.error(
                "Framework %s is not supported. Supported frameworks are %s.",
                config.extra_args.get("framework"),
                ", ".join([framework.value for framework in OVOptimumFramework]),
            )
            return False

        # validate quantization weight_format if provided
        if (
            config.ov_quant_config
            and config.ov_quant_config.get("weight_format") is not None
            and config.ov_quant_config.get("weight_format")
            not in [weight_format.value for weight_format in OVWeightFormat]
        ):
            logger.error(
                "Weight format %s is not supported. Supported weight formats are %s.",
                config.ov_quant_config.get("weight_format"),
                ", ".join([weight_format.value for weight_format in OVWeightFormat]),
            )
            return False

        # validate quantization quant_mode if provided
        if (
            config.ov_quant_config
            and config.ov_quant_config.get("quant_mode") is not None
            and config.ov_quant_config.get("quant_mode") not in [quant_mode.value for quant_mode in OVQuantMode]
        ):
            logger.error(
                "Quant mode %s is not supported. Supported quant modes are %s.",
                config.ov_quant_config.get("quant_mode"),
                ", ".join([quant_mode.value for quant_mode in OVQuantMode]),
            )
            return False

        # validate backup precisions if provided
        if (
            config.ov_quant_config
            and config.ov_quant_config.get("backup_precision") is not None
            and config.ov_quant_config.get("backup_precision")
            not in [backupmode.value for backupmode in nncf.BackupMode]
        ):
            logger.error(
                "Backup precision %s is not supported. Supported backup precisions are %s.",
                config.ov_quant_config.get("backup_precision"),
                ", ".join([backupmode.value for backupmode in nncf.BackupMode]),
            )
            return False

        return True

    def _run_for_config(
        self, model: HfModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> Union[OpenVINOModelHandler, CompositeModelHandler]:
        try:
            from optimum.exporters.openvino import main_export as export_optimum_intel
            from optimum.intel.openvino.configuration import (
                OVConfig,
                _GPTOSSQuantizationConfig,
                get_default_quantization_config,
            )
            from optimum.intel.utils.import_utils import is_nncf_available
        except ImportError as e:
            raise ImportError("Please install Intel® optimum[openvino] to use OpenVINO Optimum Conversion") from e

        # import the right quantization config depending on optimum-intel version
        try:
            from optimum.intel.openvino.configuration import _DEFAULT_4BIT_WQ_CONFIG as WRAPPER_4_BIT
        except ImportError as _:
            # fallback to older version
            logger.warning("falling back to older version of optimum-intel import API.")
            from optimum.intel.openvino.configuration import _DEFAULT_4BIT_CONFIG as WRAPPER_4_BIT

        extra_args = deepcopy(config.extra_args) if config.extra_args else {}
        extra_args.update(
            {
                "device": config.device,
            }
        )

        if extra_args.get("library") is None:
            lib_name = infer_library_name(model.model_name_or_path)
        else:
            lib_name = extra_args["library"]

        if config.ov_quant_config:
            if config.ov_quant_config.get("weight_format") is None and config.ov_quant_config.get("quant_mode") is None:
                ov_config = None
                if (
                    not no_compression_parameter_provided(config.ov_quant_config)
                    or config.ov_quant_config.get("quantization_statistics_path", None) is not None
                ):
                    raise ValueError(
                        "Some compression parameters are provided, but the weight format is not specified. "
                        "Please provide it with weight_format key in ov_quant_config dictionary."
                    )
                if not no_quantization_parameter_provided(config.ov_quant_config):
                    raise ValueError(
                        "Some quantization parameters are provided, but the quant mode is not specified. "
                        "Please provide it with quant_mode key in ov_quant_config dictionary."
                    )
            elif config.ov_quant_config.get("weight_format") in {"fp16", "fp32"}:
                ov_config = OVConfig(dtype=config.ov_quant_config["weight_format"])
            else:
                if not is_nncf_available():
                    raise ImportError("Please install nncf to use OpenVINO Optimum Conversion with quantization.")
                if (
                    config.ov_quant_config.get("weight_format") is not None
                    and config.ov_quant_config.get("quant_mode") is not None
                ):
                    # both are provided, so raise ValueError
                    raise ValueError("Both weight_format and quant_mode are provided. Please provide only one of them.")

                default_quantization_config = get_default_quantization_config(
                    model.model_name_or_path,
                    config.ov_quant_config.get("weight_format"),
                    config.ov_quant_config.get("quant_mode"),
                )

                if config.ov_quant_config.get("weight_format") is not None:
                    # weight compression
                    quant_config = prep_wc_config(config.ov_quant_config, WRAPPER_4_BIT)
                    if no_compression_parameter_provided(config.ov_quant_config) and config.ov_quant_config.get(
                        "weight_format"
                    ) in ["int4", "int8"]:
                        if default_quantization_config is not None:
                            quant_config = default_quantization_config
                            logger.info(
                                "Applying the default quantization config for model %s: %s",
                                model.model_name_or_path,
                                quant_config,
                            )
                        elif config.ov_quant_config.get("weight_format") == "int4":
                            quant_config = WRAPPER_4_BIT
                            logger.info(
                                "Applying a default 4-bit weight compression config for model %s: %s",
                                model.model_name_or_path,
                                quant_config,
                            )
                        if config.ov_quant_config.get("quantization_statistics_path", None) is not None:
                            quant_config["statistics_path"] = config.ov_quant_config.get("quantization_statistics_path")
                else:
                    if (
                        no_quantization_parameter_provided(config.ov_quant_config)
                        and default_quantization_config is not None
                    ):
                        quant_config = default_quantization_config
                        logger.info(
                            "Applying the default quantization config for model %s: %s",
                            model.model_name_or_path,
                            quant_config,
                        )
                    else:
                        if quant_config.get("dataset", None) is None:
                            raise ValueError(
                                "Dataset is required for full quantization. "
                                "Please provide it in ov_quant_config dictionary under 'dataset' key"
                            )
                        if config.ov_quant_config.get("quant_mode") in [
                            "cb4_f8e4m3",
                            "int4_f8e4m3",
                            "int4_f8e5m2",
                        ]:
                            if lib_name == "diffusers":
                                raise NotImplementedError("Mixed precision quantization isn't supported for diffusers.")
                            wc_config = prep_wc_config(config.ov_quant_config, WRAPPER_4_BIT)
                            wc_dtype, q_dtype = config.ov_quant_config["quant_mode"].split("_")
                            wc_config["dtype"] = wc_dtype

                            q_config = prep_q_config(config.ov_quant_config)
                            q_config["dtype"] = q_dtype

                            quant_config = {
                                "weight_quantization_config": wc_config,
                                "full_quantization_config": q_config,
                                "num_samples": config.ov_quant_config.get("num_samples"),
                                "dataset": config.ov_quant_config.get("dataset"),
                            }
                        else:
                            if config.ov_quant_config.get("quantization_statistics_path", None) is not None:
                                logger.warning(
                                    "quantization_statistics_path is only applicable for weight-only"
                                    " quantization. It will be ignored."
                                )
                            quant_config = prep_q_config(config.ov_quant_config)

                ov_config = OVConfig(quantization_config=quant_config)
        else:
            ov_config = None

        # quantization config
        quant_config = ov_config.quantization_config if ov_config else None

        apply_main_quantize = quant_config and not isinstance(quant_config, _GPTOSSQuantizationConfig)

        try:
            extra_args["ov_config"] = ov_config
            extra_args["stateful"] = not extra_args.get("disable_stateful", False)
            extra_args.pop("disable_stateful", False)
            extra_args["convert_tokenizer"] = not extra_args.get("disable_convert_tokenizer", False)
            extra_args.pop("disable_convert_tokenizer", False)
            extra_args["library_name"] = lib_name
            extra_args.pop("library", None)
            export_optimum_intel(
                model.model_name_or_path,
                output_model_path,
                **extra_args,
            )
            if apply_main_quantize:
                _main_quantize(
                    model_name_or_path=model.model_name_or_path,
                    task=extra_args.get("task", "auto"),
                    library_name=lib_name,
                    quantization_config=quant_config,
                    output=Path(output_model_path),
                    cache_dir=config.ov_quant_config.get("cache_dir", None) if config.ov_quant_config else None,
                    trust_remote_code=config.ov_quant_config.get("trust_remote_code", False)
                    if config.ov_quant_config
                    else False,
                    model_kwargs=model.load_kwargs.__dict__ if model.load_kwargs else None,
                )
        except Exception as e:
            raise RuntimeError(f"OpenVINO optimum export failed: {e}") from None

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


def prep_wc_config(quant_cfg, default_cfg):
    """Prepare the weight compression config for OpenVINO."""
    is_int8 = quant_cfg.get("weight_format") == "int8"
    return {
        "bits": 8 if is_int8 else 4,
        "ratio": 1.0 if is_int8 else (quant_cfg.get("ratio") or default_cfg.get("ratio")),
        "sym": quant_cfg.get("sym", False),
        "group_size": -1 if is_int8 else quant_cfg.get("group_size"),
        "all_layers": None if is_int8 else quant_cfg.get("all_layers", False),
        "dataset": quant_cfg.get("dataset"),
        "num_samples": quant_cfg.get("num_samples"),
        "quant_method": "awq" if quant_cfg.get("awq", False) else "default",
        "sensitivity_metric": quant_cfg.get("sensitivity_metric"),
        "scale_estimation": quant_cfg.get("scale_estimation", None),
        "gptq": quant_cfg.get("gptq", None),
        "lora_correction": quant_cfg.get("lora_correction", None),
        "dtype": quant_cfg.get("weight_format"),
        "backup_precision": quant_cfg.get("backup_precision"),
        "statistics_path": quant_cfg.get("statistics_path", None),
        "group_size_fallback": quant_cfg.get("group_size_fallback", None),
    }


def prep_q_config(quant_cfg):
    """Prepare the quantization config for OpenVINO."""
    return {
        "dtype": quant_cfg.get("quant_mode"),
        "bits": 8,
        "sym": quant_cfg.get("sym", False),
        "dataset": quant_cfg.get("dataset"),
        "num_samples": quant_cfg.get("num_samples"),
        "smooth_quant_alpha": quant_cfg.get("smooth_quant_alpha"),
    }


def no_compression_parameter_provided(q_config):
    return all(
        it is None
        for it in (
            q_config.get("ratio", None),
            q_config.get("group_size", None),
            q_config.get("sym", None),
            q_config.get("all_layers", None),
            q_config.get("dataset", None),
            q_config.get("num_samples", None),
            q_config.get("awq", None),
            q_config.get("scale_estimation", None),
            q_config.get("gptq", None),
            q_config.get("lora_correction", None),
            q_config.get("sensitivity_metric", None),
            q_config.get("backup_precision", None),
        )
    )


def no_quantization_parameter_provided(q_config):
    return all(
        it is None
        for it in (
            q_config.get("sym", None),
            q_config.get("dataset", None),
            q_config.get("num_samples", None),
            q_config.get("smooth_quant_alpha", None),
        )
    )
